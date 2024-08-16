import math
import torch
import torch.nn as nn
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

class CriticMoG(nn.Module):
    def __init__(self, obs_space, num_gaussians, hidden_layer_dims = None, num_layers = None, activation = None):
        super(CriticMoG, self).__init__()
        nn.Module.__init__(self)
        self.elu = torch.nn.ELU()
        self.num_gaussians = num_gaussians 
        self.activation_fn = activation if activation is not None else 'LeakyReLU'
        self.num_layers = num_layers if num_layers is not None else 2
        self.hidden_layer_dims = hidden_layer_dims if hidden_layer_dims is not None else 128
        
        if self.activation_fn in activation_functions:
            self.activation_fn = activation_functions[self.activation_fn]()
        
        layers = []
        for i in range(num_layers):
            input_dim = obs_space.shape[0]
            in_features = input_dim if i == 0 else hidden_layer_dims
            layers.append(nn.Linear(in_features, hidden_layer_dims))
            layers.append(self.activation_fn)
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_layer_dims, self.num_gaussians * 3)

    def forward(self,  input_dict, state, seq_lens):
        obs_raw = input_dict['obs_flat'].float()
        obs = obs_raw.reshape(obs_raw.shape[0], -1)
        logits = self.hidden_layers(obs)
        value_output = self.output_layer(logits)
        # get gaussians components
        means = value_output[:, :self.num_gaussians]
        self._u = means
        
        sigmas_prev = value_output[:, self.num_gaussians:self.num_gaussians*2]
        sigmas = torch.nn.functional.softplus(sigmas_prev) + 1e-6
        self._sigmas = sigmas
        
        alphas = value_output[:, self.num_gaussians*2:]
        alphas = torch.clamp(torch.nn.functional.softmax(alphas, dim=-1), 1e-6, None)
        self._alphas = alphas

        return value_output, state
    
    def value_function(self, means = None, alphas = None):
        # values of the forward pass is simply the gaussian means multiplied by their respective alpha
        # give the user the option to pass means and alphas so they have the ability to graph, etc.
        if means is not None and alphas is not None:
            value = torch.sum(means * alphas, dim = 1)
        else:
            value = torch.sum(self._u * self._alphas, dim = 1)
        return value

    def predict_gmm_params(self, obs):
        logits = self.hidden_layers(obs)
        value_output = self.output_layer(logits)
        # get gaussians components
        means = value_output[:, :self.num_gaussians]

        sigmas_prev = value_output[:, self.num_gaussians:self.num_gaussians*2]
        sigmas = torch.nn.functional.softplus(sigmas_prev) + 1e-6

        alphas = value_output[:, self.num_gaussians*2:]
        # run through softmax later since we do the logsumexp

        return means, sigmas, alphas
    

    def compute_log_likelihood(self, td_targets, mu_current, sigma_current, alpha_current):
        td_targets_expanded = td_targets.unsqueeze(1)
        sigma_clamped = sigma_current
        log_2_pi = torch.log(2*torch.tensor(math.pi))
        factor = -torch.log(sigma_clamped) - 0.5*log_2_pi 
        mus = td_targets_expanded - mu_current
        
        logp = torch.clamp(factor - torch.square(mus)/ (2*torch.square(sigma_clamped)), -1e10, 10)
        loga = torch.clamp(torch.nn.functional.log_softmax(alpha_current, dim=-1), 1e-6, None)
        
        summing_log = -torch.logsumexp(logp + loga, dim=-1)
        return summing_log

    def custom_loss(self, sample_batch, gamma = None):
        gamma = gamma if gamma is not None else 0.99
        cur_obs = sample_batch[SampleBatch.CUR_OBS]
        next_obs = sample_batch[SampleBatch.NEXT_OBS]
        rewards = sample_batch[SampleBatch.REWARDS]
        dones = sample_batch[SampleBatch.DONES]

        mu_current, sigma_current, alpha_current = self.predict_gmm_params(cur_obs)
        mu_next, sigma_next, alpha_next = self.predict_gmm_params(next_obs)
        alpha_next = torch.clamp(torch.nn.functional.softmax(alpha_next, dim=-1), 1e-6, None)

        next_state_values = torch.sum(mu_next * alpha_next, dim=1).clone().detach()
        td_targets = rewards + gamma * next_state_values * (1 - dones.float())
        
        log_likelihood = self.compute_log_likelihood(td_targets, mu_current, sigma_current, alpha_current)
        log_likelihood = torch.clamp(log_likelihood, -10, 80)
        nll_loss = torch.mean(log_likelihood)

        return nll_loss
