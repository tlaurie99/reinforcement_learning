import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ray.rllib.models.torch.misc import SlimFC

activation_functions = {
    'relu': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'elu': nn.ELU
}

class ENNWrapper(nn.Module):
    def __init__(self, base_network, z_dim, activation_name, enn_layer, hidden_layer):
        super(ENNWrapper, self).__init__()
        self.std = 1.0
        self.mean = 0.0
        self.z_dim = z_dim
        self.step_number = 0
        self.z_indices = None
        self.step_cut_off = 200

        if activation_name in activation_functions:
            activation = activation_functions[activation_name]()
        else:
            raise ValueError("Unsupported activation function")
        
        self.distribution = Normal(torch.full((self.z_dim,), self.mean), 
                                   torch.full((self.z_dim,), self.std))
        


        # Collect all layers including those inside SlimFC
        def collect_layers(module):
            layers = []
            for m in module.children():
                if isinstance(m, SlimFC):
                    layers.extend(list(m._model.children()))
                elif isinstance(m, nn.Sequential):
                    layers.extend(collect_layers(m))
                else:
                    layers.append(m)
            return layers

        # Collect the layers from the base network
        hidden_layers = collect_layers(base_network._hidden_layers)
        if base_network._logits:
            last_layer = list(base_network._logits.children())
        else:
            last_layer = []

        # Create a new sequential model with the hidden layers followed by the last layer
        self.base_network = nn.Sequential(*hidden_layers)
        self.last_layer = nn.Sequential(*last_layer)

        self.learnable_layers = nn.Sequential(
            nn.Linear(hidden_layer + 1, enn_layer), activation,
            nn.Linear(enn_layer, enn_layer, activation),
            nn.Linear(enn_layer, 1)
        )
        self.prior_layers = nn.Sequential(
            nn.Linear(hidden_layer + 1, enn_layer), activation,
            nn.Linear(enn_layer, enn_layer, activation),
            nn.Linear(enn_layer, 1)
        )

    def forward(self, obs):
        # get intermediate logits (second before last layer)
        intermediate = self.base_network(obs)
        # detach both so gradients only flow into the ENN
        base_output = self.last_layer(intermediate).detach()
        intermediate_unsqueeze = torch.unsqueeze(intermediate, 1).detach()
        # draw sample from distribution and cat to logits
        self.z_samples = self.distribution.sample((obs.size(0),)).unsqueeze(-1).to(obs.device)
        enn_input = torch.cat((self.z_samples, intermediate_unsqueeze.expand(-1, self.z_dim, -1)), dim=2)
        # enn, prior and base network pass
        if self.step_number < self.step_cut_off:
            # only updated prior for xx timesteps
            prior_out = self.prior_layers(enn_input)
        else:
            with torch.no_grad():
                # this now encapsulates the uncertainty and will inject into each timestep
                prior_out = self.prior_layers(enn_input)
        prior_bmm = torch.bmm(torch.transpose(prior_out, 1, 2), self.z_samples)
        prior = prior_bmm.squeeze(-1)
        # pass through learnable part of the ENN
        learnable_out = self.learnable_layers(enn_input)
        learnable_bmm = torch.bmm(torch.transpose(learnable_out, 1, 2), self.z_samples)
        learnable = learnable_bmm.squeeze(-1)
        enn_out = torch.mean(learnable + prior, dim = -1)
        return base_output, enn_out

    def enn_loss(self, next_obs, rewards, dones, current_critic_value, next_critic_value, gamma = None):
        gamma = gamma if gamma is not None else 0.99
        # repeating the same process as the forward pass to build the TD target
        intermediate = self.base_network(next_obs)
        intermediate_unsqueeze = torch.unsqueeze(intermediate, 1).detach()
        enn_input = torch.cat((self.z_samples, intermediate_unsqueeze.expand(-1, self.z_dim, -1)), dim=2)
        # prior
        if self.step_number < self.step_cut_off:
            prior_out = self.prior_layers(enn_input)
        else:
            with torch.no_grad():
                prior_out = self.prior_layers(enn_input)
        prior_bmm = torch.bmm(torch.transpose(prior_out, 1, 2), self.z_samples)
        prior = prior_bmm.squeeze(-1)
        # learnable
        learnable_out = self.learnable_layers(enn_input)
        learnable_bmm = torch.bmm(torch.transpose(learnable_out, 1, 2), self.z_samples)
        learnable = learnable_bmm.squeeze(-1)
        # build TD target and detach / calculate MSE
        enn_target = torch.mean(learnable + prior, dim = -1)
        next_values = next_critic_value + enn_target
        target = rewards + gamma * next_values.clone().detach() * (1 - dones.float())
        difference = torch.square(current_critic_value - target)
        mse = torch.mean(difference)
        return mse

