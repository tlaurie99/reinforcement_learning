import copy
import logging
import gymnasium
import numpy as np
import pandas as pd
import flatten_dict
from abc import abstractmethod
from gymnasium.spaces import Box
from collections import defaultdict
from ray.rllib.policy import Policy
from ray.rllib.evaluation import Episode
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.models.action_dist import ActionDistribution
from pydantic import ConfigDict, model_validator, validator
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union
from ray.rllib.policy.torch_mixins import (EntropyCoeffSchedule,KLCoeffMixin,LearningRateSchedule,ValueNetworkMixin)
from ray.rllib.utils.torch_utils import (apply_grad_clipping,explained_variance,sequence_mask,warn_if_infinite_kl_divergence)
from ray.rllib.utils.annotations import (DeveloperAPI, OverrideToImplementCustomLogic, OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)

torch, nn = try_import_torch()



class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model"""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function
            )
        else:
            self.compute_central_vf = self.model.central_value_function


def align_batch(one_opponent_batch, sample_batch):
    '''Aligns the opponents batches to each other since some may last longer compared to others by padding their observations with zeros
    Args:
        one_opponent_batch: Goes through all the other agents from the one being processed and aligns them to the sample_batch (current agent)
        sample_batch: The sample batch of data for the current agent       

    Returns:
        The non-current agents' batches aligned to that of the sample_batch
        i.e.:
           len(sample_batch['obs']):                   ------------------------            ------------------------
           len(one_opponent_batch['agent_2']['obs']):  --------------               ====>  ------------------------
           len(one_opponent_batch['agent_3']['obs']):  ------------------                  ------------------------
    '''
    length_diff = abs(len(sample_batch) - len(one_opponent_batch))
    if length_diff == 0:
        return one_opponent_batch
    elif len(one_opponent_batch) > len(sample_batch):
        one_opponent_batch = one_opponent_batch.slice(0, len(sample_batch))
    else:
        start_index = max(0, len(one_opponent_batch) - length_diff)
        padding = one_opponent_batch.slice(start_index, len(one_opponent_batch))
        for _ in range(length_diff // len(padding)):
            one_opponent_batch = one_opponent_batch.concat(padding)
        remainder = length_diff % len(padding)
        if remainder > 0:
            one_opponent_batch = one_opponent_batch.concat(padding.slice(0, remainder))        
    return one_opponent_batch

def centralized_critic_postprocessing(policy, sample_batch, config, other_agent_batches = None, episode = None):
    '''IMPORTANT: we have to update the VF preds in the sample batch so that advantages are calculated correctly

    Args:
        policy:              --The policy to compute the central value prediction for
        sample_batch:        --The sample batch of data
        config:              --The config set from local yamls
        other_agent_batches: --The sample batch of data in multi-agent scenarios which contains data for the opposite agent that is being processed        

    Returns:
        Training batch with computed advantages, vf predictions, and vf targets
        Also updates the batch to have the framestacked version for the forward pass
    '''
    custom_config = policy.config['model']['custom_model_config']
    global_state_flag = custom_config['global_state_flag']
    opp_action_in_cc = custom_config['opp_action_in_cc']
    agent_names = policy.config['policies_to_train']
    num_agents = custom_config['num_agents']
    num_frames = custom_config['num_frames']
    gamma = custom_config['gamma']
    opponent_agents_num = num_agents - 1
    
    

    if policy.loss_initialized():
        if not opp_action_in_cc and global_state_flag:
            sample_batch['all_observations'] = sample_batch['obs']
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf((convert_to_torch_tensor(sample_batch['all_observations'], policy.device))).cpu().detach().numpy()
        else:
            assert other_agent_batches is not None
            raw_list = list(other_agent_batches.values())
            if policy.config["enable_connectors"]:
                # other_agent_batches sends agent name, policy, sample_batch
                # we want the sample batches from the current agent and all other agents (in other_agent_batches)
                raw_opponent_batch = [raw_list[i][2] for i in range(opponent_agents_num)]
            else:
                raw_opponent_batch = [raw_list[i][1] for i in range(opponent_agents_num)]

            opponent_batch = []
            for one_opponent_batch in raw_opponent_batch:
                one_opponent_batch = align_batch(one_opponent_batch, sample_batch)
                opponent_batch.append(one_opponent_batch)

            if global_state_flag:
                sample_batch['all_observations'] = sample_batch['obs']
                # sample_batch['next_state'] = sample_batch['new_obs']
                sample_batch['all_actions'] = sample_batch['actions']
            else:
                state_batch_list = []
                next_state_batch_list = []
                next_immediate_obs = []
                action_list = []
                for agent_name in agent_names:
                    if agent_name in other_agent_batches:
                        index = list(other_agent_batches).index(agent_name)
                        state_batch_list.append(opponent_batch[index]['all_observations'])
                        action_list.append(opponent_batch[index]['all_actions'])
                        next_immediate_obs.append(opponent_batch[index]['new_obs'])
                    else:
                        state_batch_list.append(sample_batch['all_observations'])
                        next_immediate_obs.append(sample_batch['new_obs'])
                        action_list.append(sample_batch['all_actions'])
                '''
                This part of the processing is creating a global current state and a global next state
                These are used to predict the MoGs of the value network and compute advantages

                Flow: current_observation : prev_action ---> next_observation : current_action
                For framestacking, the offsetting is done using the num_frames specified

                The below is an example of the timesteps given num_frames = 5
                current_observation:           (t = -4, t = -3, t = -2, t = -1, t = 0)
                prev_action:                   (t = -5, t = -4, t = -3, t = -2, t = -1)

                next_observation:              (t = -3, t = -2, t = -1, t = 0, concatenated t = 1)
                current_action:                (t = -4, t = -3, t = -2, t = -1, t = 0)
                
                '''
                actions = np.stack(action_list, 1)
                actions_stacked = actions[:, :, :-1, :]
                next_actions_stacked = actions[:, :, 1:, :]
                actions_stacked = actions_stacked.reshape(actions_stacked.shape[0], -1)
                next_actions_stacked = next_actions_stacked.reshape(next_actions_stacked.shape[0], -1)

                observations = np.stack(state_batch_list, 1)
                stacked_observations = observations.reshape(observations.shape[0], -1)
                sample_batch['GlobalStateStack'] = np.concatenate((stacked_observations, actions_stacked), axis = -1)

                #process for the next observations stacking
                #since observations is (num_frames -1) already, the next observation will be 1:
                next_observations = observations[:, :, 1:, :]
                next_observations = next_observations.reshape(next_observations.shape[0], -1)
                next_immediate_obs = np.stack(next_immediate_obs, 1)
                next_immediate_obs_flat = next_immediate_obs.reshape(next_immediate_obs.shape[0], -1)
                #concatenate next observations with immediate next observation (i.e.: t = -3, t = -2, t = -1, t = 0, concatenated t = 1)
                concatenated_next_obs = np.concatenate((next_observations, next_immediate_obs_flat), axis = -1)
                #concatenate stacked actions onto the stacked next observations
                sample_batch['GlobalNextStateStack'] = np.concatenate((concatenated_next_obs, next_actions_stacked), axis = -1)
                
            #now use the central value function to get the MoG predictions
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf((convert_to_torch_tensor(sample_batch['GlobalStateStack'],   
                                                    policy.device))).cpu().detach().numpy()
    else:
        # Policy hasn't been initialized yet, use zeros
        o = sample_batch[SampleBatch.CUR_OBS]
        a = sample_batch[SampleBatch.ACTIONS]
        n_o = sample_batch[SampleBatch.NEXT_OBS]
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype = np.float32)
        sample_batch['all_actions'] = np.zeros((a.shape[0], a.shape[-1]), dtype = sample_batch[SampleBatch.ACTIONS].dtype)
        sample_batch['all_observations'] = np.zeros((o.shape[0], o.shape[-1]), dtype = sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch['GlobalStateStack'] = np.zeros((o.shape[0], (o.shape[-1] + a.shape[-1]) * num_agents * num_frames), dtype = 
                                                    sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch['GlobalNextStateStack'] = np.zeros((o.shape[0], (o.shape[-1] + a.shape[-1]) * num_agents * num_frames), dtype = 
                                                    sample_batch[SampleBatch.CUR_OBS].dtype)
        

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    #compute advantages with updated VF_PREDS
    train_batch = compute_advantages(rollout=sample_batch, last_r=last_r, gamma=gamma, lambda_=policy.config['lambda'], use_gae=config['use_gae'], 
                                    use_critic=config['use_critic'])

    return train_batch


def loss_with_central_critic(policy, model, dist_class, train_batch, config):

    '''Compute the loss using of the centralized critic
    Args:
        policy:      --The policy to use for computing the surrogate loss
        model:       --The model to use for computing the value function loss
        dist_class:  --The action distr. class
        train_batch: --The training data
        config:      --The config set from local yamls

    Returns:
        The PPO loss tensor given the input batch

    ---------------------------------------------------------------CAUTION---------------------------------------------------------------

    ****SURROGATE LOSS****
    --The base_ppo_loss function turns the surrogate loss into a NEGATIVE value
    --This is because we are maximizing the objective function which is the expected return
    --So, therefore, we are encouraging taking actions that give a higher expected return compared to baseline

    ****VALUE FUNCTION LOSS****
    --The vf_loss is POSITIVE since we are minimizing the distance between the predicted gaussian params and the target gaussian params
    --i.e. the distance between something is always positive and we are minimizing it
    
    ****MAIN POINT****
    TLDR: the part of the loss that invokes actions to get a higher return (optimization is to maximize this) is negative and the part of the loss that is 
    an error between predicted and actual is a positive in which we aim to minimize this
    '''

    # Save original value function -- this funky trick is needed for some reason (doesn't work without it)
    vf_saved = model.value_function
    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch['GlobalStateStack'],
    )
    policy._central_value_out = model.value_function()
    #calculate surrogate loss with no vf loss (this is done by the custom model)
    surrogate_loss = base_ppo_loss(model, dist_class, train_batch, config, policy)
    #compute total loss with model (this adds the surrogate loss with the vf loss to return a scalar)
    total_loss = model.custom_loss(surrogate_loss, train_batch)
    # Restore original value function.
    model.value_function = vf_saved
    #add metrics to the policy object for the stats_fn later
    policy._total_loss = total_loss
    policy._vf_loss = (policy._total_loss - surrogate_loss)

    return total_loss

def base_ppo_loss(model, dist_class, train_batch, config, policy) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Proximal Policy Objective

    Args:
        model:       --The Model to calculate the loss for
        dist_class:  --The action distr. class
        train_batch: --The training data
        config:      --The config set from local yamls
        policy:      --Policy to compute the surrogate loss from

    Returns:
        The PPO loss tensor given the input batch.
    """
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )
    
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
        warn_if_infinite_kl_divergence(policy, mean_kl_loss)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - config["clip_param"], 1 + config["clip_param"]
        ),
    )

    mean_policy_loss = reduce_mean_valid(
        -surrogate_loss
        - config["entropy_coeff"] * curr_entropy
    )

    if config["kl_coeff"] > 0.0:
        mean_policy_loss += config["kl_coeff"] * mean_kl_loss

    policy._mean_policy_loss = reduce_mean_valid(-surrogate_loss).detach()
    policy._mean_entropy = mean_entropy.detach()
    policy._mean_kl_loss = mean_kl_loss.detach()

    return mean_policy_loss

def central_vf_stats(policy, train_batch):
    """Compute stats given in the metrics at the end of an iteration

    Args:
        policy:      --The policy object to save the stats to
        train_batch: --The training data

    Returns:
        Stats given to the stats_fn
    """
    return convert_to_numpy({
        'total_loss': policy._total_loss,
        'policy_loss': policy._mean_policy_loss,
        'entropy': policy._mean_entropy,
        'kl': policy._mean_kl_loss,
        'vf_loss': policy._vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out)
    })



class SameStructurePolicy(ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, TorchPolicyV2):
    """Centralized critic with framestacking policy"""

    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(self, observation_space, action_space, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CentralizedValueMixin.__init__(self)
        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)
        #update view reqs so the input dict has the concatenated obs as well as the framestacked observations
        self.updating_view_reqs(observation_space, action_space, config)
        #initialize loss
        self._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs=False)
        #this true flag is needed to go through the other agent batches after policy initialization
        self._loss_initialized = True

    def updating_view_reqs(self, observation_space, action_space, config):
        """Sets up the view requirements for the forward pass call and training batches

        Arguments:
            num_frames {int}    -- The number of frames to stack
            observation_space:  -- The observation space definition
            shift               -- Shifts the respective data_col (can be a range like -4:0 to get t=-4,t=-3,t=-2,t=-1,t=0)
        """
        num_agents = config['model']['custom_model_config']['num_agents']
        num_frames = config['model']['custom_model_config']['num_frames']

        self.view_requirements['all_observations'] = ViewRequirement(
            data_col = 'obs',
            space = observation_space,
            shift = "-{}:0".format(num_frames - 1),
            used_for_compute_actions = True,
            used_for_training = True
        )
        self.view_requirements['all_actions'] = ViewRequirement(
            data_col = 'actions',
            space = action_space,
            shift = "-{}:0".format(num_frames),
            used_for_compute_actions = True,
            used_for_training = True
        )
        self.view_requirements['GlobalStateStack'] = ViewRequirement(
            data_col = 'all_observations',
            space = observation_space,
            shift = 0,
            used_for_compute_actions = True,
            used_for_training = True
        )
        self.view_requirements['GlobalNextStateStack'] = ViewRequirement(
            data_col = 'obs',
            space = observation_space,
            shift = 0,
            used_for_compute_actions = True,
            used_for_training = True
        )

    def loss_initialized(self):
        return self._loss_initialized

    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def loss(self, model: NLLModel, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        return loss_with_central_critic(self, model, dist_class, train_batch, self.config)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch):
        stats = convert_to_numpy(super().stats_fn(train_batch))
        stats.update(central_vf_stats(self, train_batch))
        return stats

    @override(TorchPolicyV2)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        #this is needed or it will have a memory leak as per RLLIB docs
        with torch.no_grad():
            return centralized_critic_postprocessing(self, sample_batch, self.config, other_agent_batches, episode)
