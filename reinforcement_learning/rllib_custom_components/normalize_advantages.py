import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.postprocessing import Postprocessing

class NormalizeAdvantagesCallback(DefaultCallbacks):
    @override(DefaultCallbacks)
    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        super().on_postprocess_trajectory(
            worker=worker,
            episode=episode,
            agent_id=agent_id,
            policy_id=policy_id,
            policies=policies,
            postprocessed_batch=postprocessed_batch,
            original_batches=original_batches,
            **kwargs
        )

        advantages = postprocessed_batch[Postprocessing.ADVANTAGES]
        if len(advantages) > 1:
            # per SB3 the mini batchsize must be > 1 and below is the normalization equation
            normalized_advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            postprocessed_batch[Postprocessing.ADVANTAGES] = normalized_advantages