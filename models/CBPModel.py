import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from lop.algos.cbp_linear import CBPLinear
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

'''
From the paper https://www.nature.com/articles/s41586-024-07711-7 this implements a continual back propagation critic network

to-do: abstract this as a wrapper for any rllib network
'''

class CBPModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.actor_fcnet = TorchFC(obs_space, action_space, action_space.shape[0]*2, model_config, name + 
                                   "_actor")
        hidden_layer_size = model_config['fcnet_hiddens'][0]
        
        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(obs_space.shape[0], hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc4 = nn.Linear(hidden_layer_size, 1)

        self.cbp1 = CBPLinear(self.fc1, self.fc2, replacement_rate=1e-4, maturity_threshold=100, init='kaiming', act_type='leaky_relu')
        self.cbp2 = CBPLinear(self.fc2, self.fc3, replacement_rate=1e-4, maturity_threshold=100, init='kaiming', act_type='leaky_relu')
        self.cbp3 = CBPLinear(self.fc3, self.fc4, replacement_rate=1e-4, maturity_threshold=100, init='kaiming', act_type='leaky_relu')

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.actor_fcnet(input_dict, state, seq_lens)
        means, log_stds = torch.chunk(logits, 2, -1)
        means_clamped = torch.clamp(means, -1, 1)
        log_stds_clamped = torch.clamp(log_stds, -10, 0)
        logits = torch.cat((means_clamped, log_stds_clamped), dim = -1)

        '''-----CBP implementation for critic network-----'''
        obs = input_dict['obs']
        x = self.act(self.fc1(obs))
        x = self.cbp1(x)
        x = self.act(self.fc2(x))
        x = self.cbp2(x)
        x = self.act(self.fc3(x))
        x = self.cbp3(x)
        # no activation on the output since this will be a scalar of value
        self.value = self.fc4(x)        
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self.value.squeeze(-1)

# register the custom model to make it available to Ray/RLlib
ModelCatalog.register_custom_model("CBPModel", CBPModel)
