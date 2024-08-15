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
        self.actor_means = TorchFC(obs_space, action_space, action_space.shape[0], model_config, name + 
                                   "_actor")
        self.log_std_init = model_config['custom_model_config'].get('log_std_init', 0)
        self.log_stds = nn.Parameter(torch.ones(action_space.shape[0]) * self.log_std_init, requires_grad = True)
        self.log_step = 0
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Get the model output
        means, _ = self.actor_means(input_dict, state, seq_lens)
        log_stds = self.log_stds.expand_as(means)
        logits = torch.cat((means, log_stds), dim = -1)
        self.value, _ = self.critic_fcnet(input_dict, state, seq_lens)
        self.log_step += 1
        
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self.value.squeeze(-1)

# Register the custom model to make it available to Ray/RLlib
ModelCatalog.register_custom_model("SimpleCustomTorchModel", SimpleCustomTorchModel)
