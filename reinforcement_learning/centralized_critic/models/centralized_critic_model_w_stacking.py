import math
import random
import pickle
import logging
import warnings
import gymnasium
import numpy as np
import pandas as pd

from gymnasium.spaces import Box
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import AppendBiasLayer, SlimFC, normc_initializer

from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType

import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCalls

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import MLPHeadConfig

from ray.train.torch import TorchTrainer


torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class NLLModelFrameStack(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        num_frames: int = 1,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        torch.autograd.set_detect_anomaly(True)

        self.num_gaussians = model_config['custom_model_config'].get('num_gaussians', 2)
        self.adder = model_config['custom_model_config'].get('adder', 1.0000001)
        self.num_agents = model_config['custom_model_config'].get('num_agents', 2)
        self.gamma = model_config['custom_model_config'].get('gamma', 0.99)
        self.seed = 123
        self.num_frames = model_config['custom_model_config'].get('num_frames', 5)
        self.vf_clipped_loss = model_config['custom_model_config'].get('vf_clipped_loss', 1)

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        #change observation space to account for the concatenation of agents and stacked observations/actions (if needed)
        if self.num_frames > 1:
            self.action_stack_dims = len(action_space) * self.num_frames * self.num_agents
            new_shape = ((obs_space.shape[0]) * self.num_agents * self.num_frames,)
            new_low = np.tile(obs_space.low, self.num_agents * self.num_frames)
            new_high = np.tile(obs_space.high, self.num_agents * self.num_frames)
        else:
            new_shape = ((obs_space.shape[0]) * self.num_agents,)
            new_low = np.tile(obs_space.low, self.num_agents)
            new_high = np.tile(obs_space.high, self.num_agents)

        new_obs_space = Box(low = new_low,
                            high = new_high,
                            shape = new_shape,
                            dtype = obs_space.dtype)
        #build the new observation space which is a function of num_agents, num_frames, and experiment used
        #this will be used for the expected dimensions into the critic network
        self._new_obs_space = new_obs_space
        self._value_in = None

        hiddens = list(model_config.get("fcnet_hiddens", []))
        post_fcnet_hiddens = list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        layers = []
        #actor network so leave at obs_space for CTDE
        prev_layer_size = int(np.product(obs_space.shape))
        self._action_logits = None

        # Create the input and layers leading up to the second from last layer
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size
        #The last layer is adjusted to be of size num_outputs, but it's a layer with activation.
        #num_outputs gets set by inheriting torch policy v2 by the model's output dimension
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # if num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation,
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._action_logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._action_logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)
        self._value_branch_separate = None

        if not self.vf_share_layers:          
            vf_layers = []
            #adjust input layer dimensions
            prev_vf_layer_size = (int(np.product(self._new_obs_space.shape)) + self.action_stack_dims)
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=self.num_gaussians*3,
            # initializer=normc_initializer(0.001),
            #no activation since this is the output layer
            activation_fn=None,
        )

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None
        # Since the observations are concatenated for the value network, we need a separate last input layer
        self._last_flat_in_value = None

    @OverrideToImplementCustomLogic
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        '''Compute the forward pass for the actor and the critic
        Args:
            input_dict: The input dictionary will contain the sample batch with the view requirements 
                determined by the policy and / or model
            state: list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens: 1d tensor holding input sequence lengths
        Returns:
            A tuple consisting of the model output tensor of size
            [BATCH, num_outputs]
        '''


        '''Actor forward pass to get action logits'''

        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        action_logits = self._action_logits(self._features) if self._action_logits else self._features
        if self.free_log_std:
            action_logits = self._append_free_log_std(action_logits)

        '''Critic forward pass which is used to get the MoG central value which will be used in the advantage 
            function as VF_PREDS'''
        # next_obs = input_dict['next_state']
        new_obs = input_dict['GlobalStateStack']
        next_stacked = input_dict['GlobalNextStateStack']
        #for some reason these will not be available within the SampleBatch if they are not used here
        #rllib gets rid of variables that are not used in the forward pass
        #we need these (next_obs and next_stacked) here to be updated for the postprocessing

        batch_size = new_obs.shape[0]
        expected_dim = (int(np.product(self._new_obs_space.shape)) + self.action_stack_dims)
        if new_obs.dim() == 3:
            new_obs = new_obs.reshape(batch_size, -1)
            if new_obs.shape[-1] < expected_dim:
                padding = torch.zeros(batch_size, expected_dim - new_obs.shape[-1])
                warnings.warn("Batch observations were padded! If this is after initialization then this is a problem!")
                if new_obs.is_cuda:
                    padding = padding.cuda(new_obs.device)
                new_obs = torch.cat([new_obs, padding], dim = -1)
        elif new_obs.dim() == 2:
            if new_obs.shape[-1] < expected_dim:
                padding = torch.zeros(batch_size, expected_dim - new_obs.shape[-1])
                if new_obs.is_cuda:
                    padding = padding.cuda(new_obs.device)
                new_obs = torch.cat([new_obs, padding], dim = -1)
        
        else:
            raise ValueError("Unexpected observation dimension")

        self._last_flat_in_value = new_obs.reshape(new_obs.shape[0], -1)
        critic_features = self._value_branch_separate(self._last_flat_in_value)
        value_output = self._value_branch(critic_features)

        assert value_output.size(1) == self.num_gaussians*3, 'output of gaussians should be 3N'

        i = self.num_gaussians
        means = value_output[:, :i]
        self._u = means

        elu = torch.nn.ELU()        
        sigmas = value_output[:, i:i*2]
        sigmas = elu(sigmas) + self.adder
        self._sigmas = sigmas
        
        alphas = value_output[:, i*2:]
        alphas = torch.nn.functional.softmax(alphas, dim=-1)
        self._alphas = alphas
        return action_logits, state

    @OverrideToImplementCustomLogic
    def value_function(self) -> TensorType:
        '''Compute the value of the critic if not using central value function
        Args:
        Returns:
            Value of the state given the observations
        '''
        assert self._features is not None, "must call forward() first"
        multiply = self._u * self._alphas
        values_out = torch.sum(multiply, dim = -1)
        return values_out

    @OverrideToImplementCustomLogic
    def central_value_function(self, global_state) -> TensorType:
        assert self._features is not None, "must call forward() first"
        '''Compute the value of the centralized critic
        Args:
            global_state: includes the observations and actions for all agents and number of frames specified
        Returns:
            Value of the state given the num_frames, num_agents and observation space
        '''
        mu_preds, _, w_preds = self.predict_gmm_params(global_state)
        w_preds_normalized = torch.nn.functional.softmax(w_preds, dim = -1)
        weighted_preds = (torch.sum(mu_preds * w_preds_normalized, dim = -1))#.cpu()
        return weighted_preds

    def predict_gmm_params(self, cur_obs):
        '''Compute the value of the critic if not using central value function
        Args:
            cur_obs: the current observation to predict the MoGs for
                --If this is for the next state observations this will be the target distribution
                --Which compares the predicted MoGs to the target MoGs (critic(observations) vs 
                    critic(next_observations))
                --This also involves the framestacked version with actions
        Returns:
            Mixture of gaussians given a set of observations
        '''
        batch_size = cur_obs.shape[0]
        expected_dim = (int(np.product(self._new_obs_space.shape)) + self.action_stack_dims)
        obs = cur_obs.reshape(batch_size, -1)
        if obs.shape[-1] < expected_dim:
            padding = torch.zeros(batch_size, expected_dim - obs.shape[-1])
            warnings.warn("Batch observations were padded! If this is after initialization then this is a problem!")
            if cur_obs.is_cuda:
                padding = padding.cuda(cur_obs.device)
            obs = torch.cat([obs, padding], dim = -1)

        critic_features = self._value_branch_separate(obs)
        value_output = self._value_branch(critic_features)

        elu = torch.nn.ELU()

        i = self.num_gaussians
        
        means = value_output[:, :i]
        
        sigmas_prev = value_output[:, i:i*2]
        sigmas = elu(sigmas_prev) + self.adder
        
        alphas = value_output[:, i*2:]
        return means, sigmas, alphas
    
    def compute_log_likelihood(self, td_targets, mu_pred, sigma_pred, alphas_pred):
        
        td_targets_expanded = td_targets.unsqueeze(1)
        
        sigma_clamped = torch.clamp(sigma_pred, 1e-9, None)
        
        log_2_pi = torch.log(2*torch.tensor(math.pi))
        
        mus = td_targets_expanded - mu_pred
        
        logp = torch.clamp(-torch.log(sigma_clamped) - .5 * log_2_pi - torch.square(mus) / (2*torch.square(sigma_clamped)), -1e9, None)
        loga = torch.nn.functional.log_softmax(alphas_pred, dim=-1)

        summing_log = -torch.logsumexp(logp + loga, dim=-1)
        
        return summing_log


    @OverrideToImplementCustomLogic
    def custom_loss(self, policy_loss, sample_batch):
        observations = sample_batch['GlobalStateStack']
        next_observations = sample_batch['GlobalNextStateStack']
        
        rewards = sample_batch[SampleBatch.REWARDS]
        dones = sample_batch[SampleBatch.DONES]

        mu_pred, sigma_pred, w_pred = self.predict_gmm_params(observations)
        mu_target, sigma_target, w_target = self.predict_gmm_params(next_observations)
        w_target = torch.nn.functional.softmax(w_target, dim = -1)

        
        next_state_value = torch.sum(mu_target * w_target, dim = 1).clone().detach()
        td_targets = rewards + self.gamma * next_state_value * (1 - dones.float())
        
        log_likelihood = self.compute_log_likelihood(td_targets, mu_pred, sigma_pred, w_pred)
        log_likelihood = torch.clamp(log_likelihood, -10, 80)
        nll_loss = torch.mean(log_likelihood)

        if isinstance(policy_loss, list):
            total_loss = [loss + self.vf_clipped_loss* nll_loss for loss in policy_loss]

        else:
            total_loss = self.vf_clipped_loss * nll_loss + policy_loss

        return total_loss