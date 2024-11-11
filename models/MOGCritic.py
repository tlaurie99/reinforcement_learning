import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from reinforcement_learning.MoG.MOG import MOG
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

class MOGCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTorchModelMOG, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        torch.autograd.set_detect_anomaly(True)

        self.gamma = 0.99
        self.activation_fn = model_config['fcnet_activation']
        # custom MOG class
        self.critic_network = MOG(obs_space = obs_space, num_gaussians = 3, 
                                        hidden_layer_dims = 256, num_layers = 2, 
                                        activation = self.activation_fn)
        self.actor_network = TorchFC(obs_space, action_space, action_space.shape[0]*2, 
                                      model_config, name + "_actor")
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # actor forward pass
        logits, _ = self.actor_fcnet(input_dict, state, seq_lens)
        means, log_stds = torch.chunk(logits, 2, -1)
        # assuming means are normalized between -1 and 1
        means_clamped = torch.clamp(means, -1, 1)
        # this is based on the means being -1 to 1 so the std_dev domain would be [0,1)
        # where exp(-10) and exp(0) would give the above domain for std_dev
        log_stds_clamped = torch.clamp(log_stds, -10, 0)
        logits = torch.cat((means_clamped, log_stds_clamped), dim = -1)
        # critic forward pass for MoG network
        self.critic_output, _ = self.critic_network(input_dict, state, seq_lens)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self.critic_network.value_function()
        # or the below
        # return self.critic_output.squeeze(-1)

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, sample_batch):
        critic_loss = self.critic_network.custom_loss(sample_batch = sample_batch, gamma=self.gamma)
        total_loss = [loss + critic_loss for loss in policy_loss]    
        return total_loss


ModelCatalog.register_custom_model("MOGCritic", MOGCritic)
