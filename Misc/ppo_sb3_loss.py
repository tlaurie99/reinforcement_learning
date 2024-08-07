import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
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

class CustomLossPolicy(PPOTorchPolicy):
    def __init__(self, obs_space, action_space, config):
        self.config = config
        self.log_std_list = []
        self.csv_file_path = "logs/log_std_list.csv"
        self.max_rows = 300_000
        # Check if the CSV file exists to determine if header needs to be written
        self.csv_initialized = os.path.exists(self.csv_file_path)
        PPOTorchPolicy.__init__(self, obs_space, action_space, config)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        # we do not have to do self.model as a passing argument since the method takes self as the first argument already
        # so therefore, we only have to pass the model without 'self'
        return self.custom_ppo_loss(model, dist_class, train_batch, self.config)


    # @override(PPOTorchPolicy)
    # def update_kl(self, sampled_kl):
    #     pass

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
        means, log_stds = torch.chunk(logits, 2, -1)
        means = means.detach()
        log_stds = log_stds.detach()
        log_stds_list = log_stds.tolist()
        self.log_std_list.extend(log_stds_list)
        
        if train_batch['agent_index'][0] == 1:
            # Check the number of rows in the CSV file
            if self.csv_initialized:
                current_rows = sum(1 for row in open(self.csv_file_path)) - 1  # Exclude header
            else:
                current_rows = 0

            if current_rows + len(self.log_std_list) > self.max_rows:
                # Clear the file if it exceeds max_rows
                self.csv_initialized = False  # Reset the flag to rewrite the header

            df = pd.DataFrame(self.log_std_list)
            if not self.csv_initialized:
                df.to_csv(self.csv_file_path, mode='w', index=False, header=True)
                self.csv_initialized = True
            else:
                df.to_csv(self.csv_file_path, mode='a', index=False, header=False)

            # Clear the list after saving to CSV to avoid duplicate entries
            self.log_std_list.clear()
            
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
        
        # Check for NaNs in logp_ratio and log values if found
        if torch.isnan(logp_ratio).any():
            print("NaN detected in logp ratio calculation")
            print("logp_ratio:", logp_ratio)
            raise ValueError("NaN detected in logp ratio calculation")

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

        assert not torch.isnan(total_loss).any(), "NaN in total loss"

        
    
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

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches = None, episode = None):
        with torch.no_grad():
            return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)
