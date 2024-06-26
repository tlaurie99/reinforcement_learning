import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

torch, nn = try_import_torch()

class SimpleCustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.critic_fcnet = TorchFC(obs_space, action_space, 1, model_config, name + "_critic")
        self.actor_fcnet = TorchFC(obs_space, action_space, action_space.shape[0]*2, model_config, name + 
                                   "_actor")
        self.log_step = 0

    def forward(self, input_dict, state, seq_lens):
        # Get the model output
        logits, _ = self.actor_fcnet(input_dict, state, seq_lens)
        self.value, _ = self.critic_fcnet(input_dict, state, seq_lens)
        self.log_step += 1
        
        return logits, state

    def value_function(self):
        return self.value.squeeze(-1)

# Register the custom model to make it available to Ray/RLlib
ModelCatalog.register_custom_model("SimpleCustomTorchModel", SimpleCustomTorchModel)