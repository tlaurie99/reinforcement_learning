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
        self.actor_fcnet = TorchFC(obs_space, action_space, action_space.shape[0]*2, model_config, name + 
                                   "_actor")
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Get the model output
        logits, _ = self.actor_fcnet(input_dict, state, seq_lens)
        means, log_stds = torch.chunk(logits, 2, -1)
        # assuming means are normalized between -1 and 1
        means_clamped = torch.clamp(means, -1, 1)
        # this is based on the means being -1 to 1 so the std_dev domain would be [0,1)
        # where exp(-10) and exp(0) would give the above domain for std_dev
        log_stds_clamped = torch.clamp(log_stds, -10, 0)
        logits = torch.cat((means_clamped, log_stds_clamped), dim = -1)
        self.value, _ = self.critic_fcnet(input_dict, state, seq_lens)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self.value.squeeze(-1)

# register the custom model to make it available to Ray/RLlib
ModelCatalog.register_custom_model("SimpleCustomTorchModel", SimpleCustomTorchModel)
