import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ray.rllib.models.torch.misc import SlimFC

activation_functions = {
    'Threshold': nn.Threshold,
    'ReLU': nn.ReLU,
    'RReLU': nn.RReLU,
    'Hardtanh': nn.Hardtanh,
    'ReLU6': nn.ReLU6,
    'Sigmoid': nn.Sigmoid,
    'Hardsigmoid': nn.Hardsigmoid,
    'Tanh': nn.Tanh,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish,
    'Hardswish': nn.Hardswish,
    'ELU': nn.ELU,
    'CELU': nn.CELU,
    'SELU': nn.SELU,
    'GLU': nn.GLU,
    'GELU': nn.GELU,
    'Hardshrink': nn.Hardshrink,
    'LeakyReLU': nn.LeakyReLU,
    'LogSigmoid': nn.LogSigmoid,
    'Softplus': nn.Softplus,
    'Softshrink': nn.Softshrink,
    'MultiheadAttention': nn.MultiheadAttention,
    'PReLU': nn.PReLU,
    'Softsign': nn.Softsign,
    'Tanhshrink': nn.Tanhshrink,
    'Softmin': nn.Softmin,
    'Softmax': nn.Softmax,
    'Softmax2d': nn.Softmax2d,
    'LogSoftmax': nn.LogSoftmax,
}

class ENNWrapper(nn.Module):
    def __init__(self, base_network, z_dim, enn_layer, activation = None, initializer = None):
        super(ENNWrapper, self).__init__()
        """
        Args:
            base_network: network that is wrapped with the ENN
            z_dim: number of dimensions for the multivariate gaussian distribution
                -- This can be seen as the number of models (mimicking the ensemble approach with noise)
            enn_layer: layer size for the enn
            hidden_layer: base network layer size
            activation: activation function to use for the base and enn networks
            initializer: network initializer to use
                -- Recommended to leave default per https://arxiv.org/abs/2302.09205
        """
        self.std = 1.0
        self.mean = 0.0
        self.z_dim = z_dim
        self.step_number = 0
        self.z_indices = None
        self.step_cut_off = 100
        self.activation_fn = activation if activation is not None else 'LeakyReLU'
        self.initializer = initializer if initializer is not None else torch.nn.init.xavier_normal_
        self.distribution = Normal(torch.full((self.z_dim,), self.mean), torch.full((self.z_dim,), self.std))

        if activation in activation_functions:
            activation = activation_functions[activation]()
        else:
            raise ValueError("Unsupported activation function")
            
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
        
        def get_last_layer_input_features(layers):
            for layer in reversed(layers):
                if isinstance(layer, nn.Linear):
                    return layer.in_features
            return None

        # collect the layers from the base network
        hidden_layers = collect_layers(base_network._hidden_layers)
        hidden_layer_size = get_last_layer_input_features(hidden_layers)
        if base_network._logits:
            last_layer = list(base_network._logits.children())
        else:
            last_layer = []

        # create a new sequential model with the hidden layers followed by the last layer
        self.base_network = nn.Sequential(*hidden_layers)
        self.last_layer = nn.Sequential(*last_layer)

        self.learnable_layers = nn.Sequential(
            SlimFC(hidden_layer_size + 1, enn_layer, initializer=self.initializer,
                   activation_fn=self.activation_fn),
            SlimFC(enn_layer, enn_layer, initializer=self.initializer, activation_fn=self.activation_fn),
            SlimFC(enn_layer, 1, initializer=self.initializer, activation_fn=self.activation_fn)
        )
        self.prior_layers = nn.Sequential(
            SlimFC(hidden_layer_size + 1, enn_layer, initializer=self.initializer,
                   activation_fn=self.activation_fn),
            SlimFC(enn_layer, enn_layer, initializer=self.initializer, activation_fn=self.activation_fn),
            SlimFC(enn_layer, 1, initializer=self.initializer, activation_fn=self.activation_fn)
        )

    def forward(self, input_dict, state, seq_lens):
        # get intermediate logits (second before last layer)
        obs_raw = input_dict['obs_flat'].float()
        obs = obs_raw.reshape(obs_raw.shape[0], -1)
        base_output, enn_out = self.pass_through_layers(obs)
        self.total_output = enn_out + base_output
        return self.total_output, state
    
    def pass_through_layers(self, obs):
        with torch.no_grad():
            intermediate = self.base_network(obs)
            base_output = self.last_layer(intermediate)
        intermediate_unsqueeze = torch.unsqueeze(intermediate, 1)
        # draw sample from distribution and cat to logits
        self.z_samples = self.distribution.sample((obs.shape[0],)).unsqueeze(-1).to(obs.device)
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
        enn_output = learnable + prior
        return base_output, enn_output

    def enn_loss(self, next_obs, rewards, dones, gamma = None):
        gamma = gamma if gamma is not None else 0.99
        next_base_output, next_enn_output = self.pass_through_layers(next_obs)
        next_values = next_base_output + next_enn_output
        next_values = next_values.squeeze(-1) if next_values.shape[-1] == 1 else next_values
        target = rewards + gamma * next_values.clone().detach() * (1 - dones.float())
        mse = torch.nn.functional.mse_loss(self.total_output.squeeze(-1), target)
        return mse

