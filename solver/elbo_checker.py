import torch
import numpy as np


def logmeanexp(x):
    assert x.dim() == 1
    return torch.logsumexp(x, 0, keepdim=False) - np.log(len(x))


def evaluate_log_p_O(rpg, env, batch_size, n_batch=1):
    # compute the prior averaged rewards ..
    log_p_o = []
    for i in range(n_batch):
        r = rpg.inference(env, prior=True, batch_size=batch_size)[
            'r'].sum(axis=0) * rpg._cfg.weight.reward
        log_p_o.append(r)
    #assert r.max() <= 1
    log_p_o = torch.concat(log_p_o)
    # print(torch.mean(torch.exp(log_p_o)).log())
    return logmeanexp(log_p_o)


def log_p_z_tau(traj):
    # from either the prior or posterior
    return traj['log_p_z'].sum(axis=0) + traj['log_p_a'].sum(axis=0)


def log_p_tau(traj):
    return traj['log_p_a'].sum(axis=0)


@torch.no_grad()
def evaluate_elbo_from_prior(rpg, env, batch_size, n_batch=1, log_p_o=None):
    # pass
    # seems that we have to fix the initial state ..
    elbos = []
    kls = []

    # TODO: note that we can only do with single state ..
    if log_p_o is None:
        log_p_o = evaluate_log_p_O(rpg, env, batch_size, n_batch)

    for i in range(n_batch):
        states, goals = env.reset(
            batch_size=batch_size, return_state_goal=True)
        assert states.std(axis=0).max() < 1e-10
        theta_tau = rpg.inference(
            env, batch_size=batch_size, states=states, goals=goals)

        traj = (theta_tau['z'], theta_tau['a'])
        prior_tau = rpg.inference(
            env, label_traj=traj, prior=True, states=states, goals=goals)
        elbo, logp_z_cond_tau, _ = rpg.elbo(**theta_tau)

        elbo = elbo.sum(axis=0)
        logp_z_cond_tau = logp_z_cond_tau.sum(axis=0)
        elbos.append(elbo)

        # p(tau|O) = p(O|tau)p(\tau) / p(O)
        logp_tau_cond_O = prior_tau['r'].sum(axis=0) * rpg._cfg.weight.reward + log_p_tau(prior_tau) - log_p_o
        logp_z_tau_cond_O = logp_tau_cond_O + logp_z_cond_tau

        kls.append(log_p_z_tau(theta_tau) - logp_z_tau_cond_O)

    elbo = torch.mean(torch.concat(elbos))
    KL = torch.mean(torch.concat(kls))
    print('log p(O)', log_p_o.item())
    print('elbo', elbo.item())
    print('KL(p_θ(z, τ)||p(z|τ)p(τ|O))', KL.item())
    print('left', log_p_o.item(), 'right', (elbo + KL).item())