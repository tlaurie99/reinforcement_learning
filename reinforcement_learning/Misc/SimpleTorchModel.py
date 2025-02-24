import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

torch, nn = try_import_torch()

class SimpleCustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.critic_fcnet = TorchFC(obs_space, action_space, 1, model_config, name + "_critic")
        self.actor_fcnet = TorchFC(obs_space, action_space, action_space.shape[0]*2, model_config, name + "_actor")
        self.log_step = 0
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Get the model output
        obs = input_dict['obs_flat']
        assert not torch.isnan(obs).any(), "nan in the observations"
        logits, _ = self.actor_fcnet(input_dict, state, seq_lens)
        self.value, _ = self.critic_fcnet(input_dict, state, seq_lens)
        self.log_step += 1

        if torch.isnan(logits).any():
            nan_indices = torch.nonzero(torch.isnan(logits), as_tuple=True)[0]
            for idx in nan_indices:
                source = 'means' if idx < logits.shape[0] // 2 else 'log_stds'
                print(f"NaN detected at index {idx} in {source}")
        
        return logits, state
        
    @override(TorchModelV2)
    def value_function(self):
        return self.value.squeeze(-1)

# Register the custom model to make it available to Ray/RLlib
ModelCatalog.register_custom_model("SimpleCustomTorchModel", SimpleCustomTorchModel)
