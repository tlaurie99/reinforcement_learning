import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from reinforcement_learning.MoG.MoG_module import CriticMoG
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

class CustomTorchModelMOG(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTorchModelMOG, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        torch.autograd.set_detect_anomaly(True)

        self.gamma = 0.99
        self.step_number = 0
        self.activation_fn = model_config['fcnet_activation']
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.critic_network = CriticMoG(obs_space = obs_space, num_gaussians = 3, 
                                        hidden_layer_dims = 256, num_layers = 2, 
                                        activation = self.activation_fn)
        self.actor_network = TorchFC(obs_space, action_space, action_space.shape[0]*2, 
                                      model_config, name + "_actor")
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs_flat'].float()
        batch_size = obs.shape[0]
        # actor forward pass
        raw_action_logits, _ = self.actor_network(input_dict, state, seq_lens)
        # critic forward pass for MoG network
        self.critic_output, _ = self.critic_network(input_dict, state, seq_lens)
        self.step_number += 1
        
        return raw_action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self.critic_network.value_function()

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, sample_batch):
        critic_loss = self.critic_network.custom_loss(sample_batch = sample_batch, gamma=self.gamma)
        total_loss = [loss + critic_loss for loss in policy_loss]
        
        if self.step_number % 1_000 == 0:
            print(f"policy loss: {policy_loss} enn loss: {total_loss}")
    
        return total_loss


ModelCatalog.register_custom_model("custom_torch_model_mog", CustomTorchModelMOG)
