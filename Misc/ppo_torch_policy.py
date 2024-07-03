import torch
import torch.nn as nn
import pandas as pd
from models.SimpleTorchModel import SimpleCustomTorchModel
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

class SimpleTorchPolicy(PPOTorchPolicy):
    def __init__(self, obs_space, action_space, config):
        self.log_step = 0
        self.log_data = pd.DataFrame(columns = ['timestep', 'logp_ratio', 'surrogate_loss', 'advantages'])
        self.parquet_file_name = config['model']['custom_model_config']['parquet_file_name']
        super().__init__(obs_space, action_space, config)
        self.model = SimpleCustomTorchModel(
            obs_space = obs_space, 
            action_space = action_space, 
            num_outputs = config['model']['custom_model_config']['num_outputs'], 
            model_config = config['model']['custom_model_config'], 
            name = 'simple_custom_torch_model')

        self.config = config

        print("initialization")
        
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward()

    def value_function(self):
        return self.model.value_function()

    def log_to_dataframe(self, logp_ratio, surrogate_loss, advantages):
        new_log_entry = pd.DataFrame({
            'timestep': [self.log_step],
            'logp_ratio': [logp_ratio.min().item() if logp_ratio is not None else np.nan],
            'surrogate_loss': [surrogate_loss.min().item() if surrogate_loss is not None else np.nan],
            'advantages': [advantages.mean().item() if advantages is not None else np.nan]
        })
    
        self.log_data = pd.concat([self.log_data, new_log_entry], ignore_index = True)
    
        if self.log_step % 1000 == 0:
            self.save_to_parquet()

    def save_to_parquet(self):
        self.log_data.to_parquet(self.parquet_file_name)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        # we do not have to do self.model as a passing argument since the method takes self as the first argument already
        # so therefore, we only have to pass the model without 'self'
        return self.custom_ppo_loss(model, dist_class, train_batch, self.config)

    def custom_ppo_loss(self, model, dist_class, train_batch, config):
        """Compute loss for Proximal Policy Objective.
    
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.
    
        Returns:
            The PPO loss tensor given the input batch.
        """
        
    
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
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
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)
    
        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)
    
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        if self.log_step is not None:
            self.log_to_dataframe(logp_ratio = logp_ratio, surrogate_loss = surrogate_loss, advantages = 
                              train_batch[Postprocessing.ADVANTAGES])
    
        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)
    
        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )
    
        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss
    
        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
    
        return total_loss
