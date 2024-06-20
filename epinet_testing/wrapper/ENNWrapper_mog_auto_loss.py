import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.policy.sample_batch import SampleBatch

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
    def __init__(self, base_network, z_dim, enn_layer, activation = None, initializer = None, 
                 using_mog_module = False):
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
        self.elu = torch.nn.ELU()
        self.using_mog_module = using_mog_module
        self.activation_fn = activation if activation is not None else 'LeakyReLU'
        self.initializer = initializer if initializer is not None else torch.nn.init.xavier_normal_
        self.distribution = Normal(torch.full((self.z_dim,), self.mean), torch.full((self.z_dim,), self.std))

        if self.activation_fn not in activation_functions:
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
        if getattr(base_network, '_hidden_layers', None):
            hidden_layers = collect_layers(base_network._hidden_layers)
            hidden_layer_size = get_last_layer_input_features(hidden_layers)
        elif getattr(base_network, 'hidden_layers', None):
            hidden_layers = collect_layers(base_network.hidden_layers)
            hidden_layer_size = get_last_layer_input_features(hidden_layers)
        else:
            raise ValueError("Unsupported base network structure")

        if getattr(base_network, '_logits', None):
            last_layer = base_network._logits
        elif getattr(base_network, 'output_layer', None):
            last_layer = base_network.output_layer
        else:
            raise ValueError("Unsupported last layer network structure")
        
        if getattr(base_network, 'activation_fn', None):
            activation_fn_base = base_network.activation_fn
        else:
            activation_fn_base = self.activation_fn

        # create a new sequential model with the hidden layers followed by the last layer
        self.base_network = nn.Sequential(*hidden_layers)
        self.last_layer = nn.Sequential(last_layer, activation_fn_base)

        self.learnable_layers = nn.Sequential(
            SlimFC(hidden_layer_size + 1, enn_layer, initializer=self.initializer,
                   activation_fn=self.activation_fn),
            SlimFC(enn_layer, enn_layer, initializer=self.initializer, activation_fn=self.activation_fn),
            SlimFC(enn_layer, 1, initializer=self.initializer, activation_fn=self.activation_fn)
        )
        self.prior_layers = nn.Sequential(
            SlimFC(hidden_layer_size + 1, enn_layer, initializer=self.initializer),
            SlimFC(enn_layer, enn_layer, initializer=self.initializer),
            SlimFC(enn_layer, 1, initializer=self.initializer)
        )

    def forward(self, input_dict, state, seq_lens):
        # get intermediate logits (second before last layer)
        obs_raw = input_dict['obs_flat'].float()
        obs = obs_raw.reshape(obs_raw.shape[0], -1)          
        self.base_output, self.enn_out = self.pass_through_layers(obs)
        
        if len(self.base_output[-1]) > 1:
            # return gaussians to user in case they want to graph, etc.
            return self.base_output, state
        else:
            # this is a simple way to keep the MoG critic and the mean critic separate
            return self.base_output + self.enn_out, state
    
    def value_function(self, means = None, alphas = None):
        # this value function does not need to be called, but can be called for a MoG network
        if means is not None and alphas is not None:
            value = torch.sum(means * alphas, dim = 1)
            total_value = self.enn_out + value
        else:
            means, sigmas, alphas = self.split_components(self.base_output)
#             i = len(self.base_output[-1]) // 3
#             means = self.base_output[:, :i]
#             alphas = self.base_output[:, i*2:]
#             alphas = torch.nn.functional.softmax(alphas, dim=-1)
            value = torch.sum(means * alphas, dim = 1)
            total_value = self.enn_out.squeeze(1) + value
        return total_value
    
    def split_components(self, components):
        i = len(components[-1]) // 3
        means = components[:, :i]
        sigmas_prev = components[:, i:i*2]
        sigmas = self.elu(sigmas_prev) + 1.000000001
        alphas = components[:, i*2:]
        alphas = torch.nn.functional.softmax(alphas, dim=-1)     
        return means, sigmas, alphas
    
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
    
    def compute_log_likelihood(self, td_targets, mu_pred, sigma_pred, alphas_pred):
        td_targets_expanded = td_targets.unsqueeze(1)
        sigma_clamped = torch.clamp(sigma_pred, 1e-9, None)
        # alphas_clamped = torch.clamp(alpha_pred, 1e-30, 1e5)
        log_2_pi = torch.log(2*torch.tensor(math.pi))
        mus = td_targets_expanded - mu_pred
        logp = torch.clamp(-torch.log(sigma_clamped) - .5 * log_2_pi - torch.square(mus) / 
                           (2*torch.square(sigma_clamped)), -1e9, None)
        loga = torch.nn.functional.log_softmax(alphas_pred, dim=-1)
        summing_log = -torch.logsumexp(logp + loga, dim=-1)

        return summing_log
        
    def enn_loss(self, sample_batch, handle_loss, gamma = None):        
        cur_obs = torch.tensor(sample_batch[SampleBatch.CUR_OBS])
        next_obs = torch.tensor(sample_batch[SampleBatch.NEXT_OBS])
        rewards = sample_batch[SampleBatch.REWARDS]
        dones = sample_batch[SampleBatch.DONES]
        gamma = gamma if gamma is not None else 0.99
        
        critic_loss = 0
        enn_loss = 0
        total_loss = 0
        
        # pass through for the enn loss if using only the mean critic
        base_output, enn_output = self.pass_through_layers(cur_obs)
        next_base_output, next_enn_output = self.pass_through_layers(next_obs)

        i = len(next_base_output[-1]) // 3
        if handle_loss:
            if i < 1:            
                base_target = rewards + self.gamma * next_base_output.clone().detach() * (1 - dones.float())
                critic_loss = torch.nn.functional.mse_loss(base_output.squeeze(-1), base_target)
            else:
                # if MoG critic, calculate loss using the negative log-likelihood
                # have to manually pass through base network if we want to allow gradients to flow
                intermediate = self.base_network(cur_obs)
                base_components = self.last_layer(intermediate)

                mu_current, sigma_current, alpha_current = self.split_components(base_components)
                mu_next, sigma_next, alpha_next = self.split_components(next_base_output)
                alpha_next = torch.nn.functional.softmax(alpha_next, dim = -1)

                next_base_output = torch.sum(mu_next * alpha_next, dim = 1).clone().detach()
                td_targets = rewards + gamma * next_base_output * (1 - dones.float())
                log_likelihood = self.compute_log_likelihood(td_targets, mu_current, sigma_current, 
                                                             alpha_current)
                log_likelihood = torch.clamp(log_likelihood, -10, 80)
                critic_loss = torch.mean(log_likelihood)

                alpha_current = torch.nn.functional.softmax(alpha_current, dim = -1)
                base_output = torch.sum(mu_current* alpha_current, dim = 1).clone().detach()
                # detach here so the gradients only will flow for the enn network below
            
        total_value_output = base_output + enn_output
        next_values = next_base_output + next_enn_output
        next_values = next_values.squeeze(-1) if next_values.shape[-1] == 1 else next_values
        target = rewards + gamma * next_values.clone().detach() * (1 - dones.float())
        enn_loss = torch.nn.functional.mse_loss(total_value_output.squeeze(-1), target)
        
        if handle_loss:
            total_loss = enn_loss + critic_loss
        else:
            total_loss = enn_loss
        
        return total_loss

