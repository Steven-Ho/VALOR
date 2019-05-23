# Networks of value / policy / decoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation, output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi

class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dims=(64, 64), activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()

        if policy is None:
            if isinstance(action_space, Box):
                self.policy = GaussianPolicy(input_dim, hidden_dims, activation, output_activation, action_space.shape[0])
            elif isinstance(action_space, Discrete):
                self.policy = CategoricalPolicy(input_dim, hidden_dims, activation, output_activation, action_space.n)
        else: 
            self.policy = policy(input_dim, hidden_dims, activation, output_activation, action_space)

        self.value_f = MLP(layers=[input_dim] + list(hidden_dims) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v= self.value_f(x)

        return pi, logp, logp_pi, v

# Bidirectional LSTM for encoding trajectories
# Batch-first used
# input: (batch_size, seq_len, input_dim) where input concatenate state and action
# inter_state: (batch_size, seq_len, 2*hidden_dims)
# linear_output: (batch_size, seq_len, context_dim)
# avg_logits: (batch_size, context_dim)
class Discriminator(nn.Module):
    def __init__(self, input_dim, context_dim, output_activation=torch.softmax, num_layers=1, hidden_dims=64):
        super(Discriminator, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dims*2, context_dim)
        nn.init.zeros_(self.linear.bias)

    def forward(self, seq):
        inter_state, _ = self.lstm(seq)
        linear_output = self.linear(inter_state)
        logits = F.softmax(linear_output, dim=-1)
        avg_logits = np.mean(logits, axis=1)

        return avg_logits