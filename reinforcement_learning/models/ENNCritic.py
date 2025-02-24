import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from reinforcement_learning.epinet_testing.wrapper.ENNWrapper_loss_handling import ENNWrapper



class ENNCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ENNCritic, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        nn.Module.__init__(self)
        torch.autograd.set_detect_anomaly(True)
        enn_layer = 50
        self.gamma = 0.99
        self.step_number = 0
        self.action_space = action_space
        self.initializer = torch.nn.init.xavier_normal_
        self.activation_fn = model_config['fcnet_activation']
        self.z_dim = model_config['custom_model_config'].get('z_dim', 5)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_critic_network = TorchFC(obs_space, action_space, 1, 
                                      model_config, name + "_critic")
        self.actor_network = TorchFC(obs_space, action_space, action_space.shape[0]*2, 
                                      model_config, name + "_actor")
        self.critic_network = ENNWrapper(base_network = self.base_critic_network, z_dim = self.z_dim, 
                                      enn_layer = enn_layer, activation = self.activation_fn, 
                                      initializer = self.initializer)
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # actor forward pass
        raw_action_logits, _ = self.actor_network(input_dict, state, seq_lens)
        # use wrapper for critic output / output gradients are blocked - detached so only to update enn
        self.critic_output, _ = self.critic_network(input_dict, state, seq_lens)
        self.step_number += 1
        
        return raw_action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self.critic_output.squeeze(-1)

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, sample_batch):
        enn_loss = self.critic_network.enn_loss(sample_batch = sample_batch, handle_loss = True, 
                                              gamma=self.gamma)
        total_loss = [loss + enn_loss for loss in policy_loss]
        return total_loss


ModelCatalog.register_custom_model("ENNCritic", ENNCritic)