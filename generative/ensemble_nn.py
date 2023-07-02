# ensemble bayesian network
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)

from scipy.stats import truncnorm
truncnorm = truncnorm(-2, 2)

def truncated_normal(size, std):
    trunc = truncnorm.rvs(size=size) * std
    return torch.tensor(trunc, dtype=torch.float32)


class ensemble_fc(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, swish=False):
        super(ensemble_fc, self).__init__()

        w = truncated_normal(size=(ensemble_size, in_features, out_features),
                             std=1.0 / (2.0 * np.sqrt(in_features)))

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))
        self.swish = swish

    def forward(self, inputs):
        # inputs (ensemble size, batch, in_feature)
        # w (ensemble size, in_feature, out_features)
        inputs = inputs.matmul(self.w) + self.b
        if self.swish:
            inputs = swish(inputs)
        return inputs


def ensemble_mlp(ensemble_size, in_features, out_features, num_layers, mid_channels):
    layers = []
    if num_layers == 1:
        layers.append(ensemble_fc(ensemble_size, in_features, out_features))
    else:
        layers.append(ensemble_fc(ensemble_size, in_features, mid_channels, swish=True))
        for i in range(num_layers-2):
            layers.append(ensemble_fc(ensemble_size, mid_channels, mid_channels, swish=True))
        layers.append(ensemble_fc(ensemble_size, mid_channels, out_features))
    return nn.Sequential(*layers)


class GaussianLayer(nn.Module):
    def __init__(self, out_features):
        super(GaussianLayer, self).__init__()

        self.out_features = out_features

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def forward(self, inputs):
        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def decay(self):
        return self.max_logvar.sum() - self.min_logvar.sum()


class EnBNN(nn.Module):
    # ensemble bayesian
    def __init__(self, ensemble_size, in_features, out_features, num_layers, mid_channels):
        super(EnBNN, self).__init__()
        self.ensemble_size = ensemble_size
        self.mlp = ensemble_mlp(ensemble_size, in_features, out_features * 2, num_layers, mid_channels)
        self.gaussian = GaussianLayer(out_features * 2)

    def forward(self, obs, action):
        # obs (ensemble, batch, dim_obs) or (batch, dim_obs)
        # action (ensemble, batch, action)
        inp = torch.cat((obs, action), dim=-1)
        if inp.shape == 2:
            inp = inp[None, :, :].expand(self.ensemble_size, -1, -1)
        return self.gaussian(self.mlp(inp))

    def var_reg(self):
        return self.gaussian.decay()

    def decay(self, weights=0.0001):
        if isinstance(weights, float):
            weights = [weights] * len(self.mlp)
        loss = 0
        for w, m in zip(weights, self.mlp):
            loss = w * (m.w ** 2).sum() / 2.0 + loss
        return loss


class EnBNNAgent(AgentBase):
    def __init__(self, lr, env, weight_decay=0.0002, var_reg=0.01, npart=20,
                 ensemble_size=5, normalizer=True, *args, **kwargs):
        extension = env.extension
        inp_dim = extension.observation_shape[0]

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.forward_model = EnBNN(ensemble_size, inp_dim + action_dim, obs_dim, *args, **kwargs)

        self.normalizer = normalizer
        self.npart = npart
        self.ensemble_size = ensemble_size
        assert self.npart % self.ensemble_size == 0 and self.npart > 0

        if self.normalizer:
            self.obs_norm: Normalizer = Normalizer((inp_dim,))
            self.action_norm: Normalizer = Normalizer((action_dim,))

        super(EnBNNAgent, self).__init__(self.forward_model, lr)
        self.weight_decay = weight_decay
        self.var_reg = var_reg

        self.extension = extension # which is actually a config file of the environment
        self.ensemble_size = ensemble_size

    def cuda(self):
        if self.normalizer:
            self.obs_norm.cuda()
            self.action_norm.cuda()
        return super(EnBNNAgent, self).cuda()

    def get_predict(self, s, a):
        inp = self.extension.encode_obs(s)
        if self.normalizer:
            inp = self.obs_norm(inp)
            a = self.action_norm(a)
        mean, log_var = self.forward_model(inp, a)
        return self.extension.add(s, mean), log_var

    def rollout(self, s, a):
        # s (inp_dim)
        # a (pop, T, acts)
        with torch.no_grad():
            if len(s.shape) == 1:
                s = s[None, :].expand(a.shape[0], -1)
            s = s[None, :].expand(self.npart, -1, -1).reshape(self.ensemble_size, -1, *s.shape[1:])

            outs = []
            rewards = 0
            for i in range(a.shape[1]):
                act = a[None, :, i].expand(self.npart, -1, -1).reshape(self.ensemble_size, -1, *a.shape[2:])
                mean, log_var = self.get_predict(s, act)
                t = torch.randn_like(log_var) * torch.exp(log_var * 0.5) + mean # sample
                outs.append(t)
                rewards = self.extension.cost(s, act, t) + rewards
                s = t
            return torch.stack(outs, dim=2), rewards.reshape(self.ensemble_size, -1, a.shape[0]).mean(dim=(0, 1))

    def rollout2(self, obs, weights):
        obs = obs.expand(weights.shape[0], -1) # (500, x)
        reward = 0
        for i in range(weights.shape[1]):
            action = weights[:, i]
            t, _ = self.forward(obs, action) # NOTE that
            if len(t.shape) == 3:
                t = t.mean(dim=0) # mean
            reward = self.extension.cost(obs, action, t) + reward
            obs = t
        return obs, reward

    def forward(self, s, a):
        return self.get_predict(s, a)

    def fit_normalizer(self, buffer):
        # TODO: very very ugly
        if self.normalizer:
            print('fit normalizer...')
            data_gen = buffer.make_sampler('fix', 'train', 1, use_tqdm=False)
            # very strange function
            idx = 0
            for s, a, _ in data_gen:
                s = self.extension.encode_obs(s)
                if idx == 0:
                    self.obs_norm.fit(s)
                    self.action_norm.fit(a)
                    idx = 1
                else:
                    self.obs_norm.update(s)
                    self.action_norm.update(a)
            print('normalizer')
            print(self.obs_norm.mean, self.obs_norm.std, self.obs_norm.count)

    def update(self, s, a, t):
        if self.training:
            self.optim.zero_grad()

        mean, log_var = self.get_predict(s, a)

        inv_var = torch.exp(-log_var)
        loss = ((mean - t.detach()[None, :]) ** 2) * inv_var + log_var
        loss = loss.mean(dim=(-2, -1)).sum(dim=0) # sum across different models

        loss += self.var_reg * self.forward_model.var_reg()
        loss += self.forward_model.decay(self.weight_decay)

        if self.training:
            loss.backward()
            self.optim.step()
        return {
            'loss': loss.detach().cpu().numpy()
        }