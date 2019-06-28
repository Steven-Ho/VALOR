# VALOR implementation 
import numpy as np 
import torch
import torch.nn.functional as F 
import gym 
import time
import scipy.signal
from network import Discriminator, ActorCritic, count_vars
from buffer import Buffer
from torch.distributions.categorical import Categorical
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
from utils.mpi_torch import average_gradients, sync_all_params
from utils.logx import EpochLogger

def valor(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), disc=Discriminator, dc_kwargs=dict(), seed=0, episodes_per_epoch=40,
        epochs=50, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, dc_lr=5e-4, train_v_iters=80, train_dc_iters=10, train_dc_interv=10, 
        lam=0.97, max_ep_len=1000, logger_kwargs=dict(), con_dim=5, save_freq=10, k=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Model
    actor_critic = actor_critic(input_dim=obs_dim[0]+con_dim, **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = Buffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)

    # Count variables
    var_counts = tuple(count_vars(module) for module in
        [actor_critic.policy, actor_critic.value_f, disc.policy])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n'%var_counts)    

    # Optimizers
    train_pi = torch.optim.Adam(actor_critic.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(actor_critic.value_f.parameters(), lr=vf_lr)
    train_dc = torch.optim.Adam(disc.policy.parameters(), lr=dc_lr)

    # Parameters Sync
    sync_all_params(actor_critic.parameters())
    sync_all_params(disc.parameters())

    def update(e):
        obs, act, adv, pos, ret, logp_old = [torch.Tensor(x) for x in buffer.retrieve_all()]
        
        # Policy
        _, logp, _ = actor_critic.policy(obs, act)
        entropy = (-logp).mean()

        # Policy loss
        pi_loss = -(logp*(k*adv+pos)).mean()

        # Train policy
        train_pi.zero_grad()
        pi_loss.backward()
        average_gradients(train_pi.param_groups)
        train_pi.step()

        # Value function
        v = actor_critic.value_f(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(train_v_iters):
            v = actor_critic.value_f(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            average_gradients(train_v.param_groups)
            train_v.step()

        # Discriminator
        if (e+1) % train_dc_interv == 0:
            print('Discriminator Update!')
            con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]
            _, logp_dc, _ = disc(s_diff, con)
            d_l_old = -logp_dc.mean()

            # Discriminator train
            for _ in range(train_dc_iters):
                _, logp_dc, _ = disc(s_diff, con)
                d_loss = -logp_dc.mean()
                train_dc.zero_grad()
                d_loss.backward()
                average_gradients(train_dc.param_groups)
                train_dc.step()

            _, logp_dc, _ = disc(s_diff, con)
            dc_l_new = -logp_dc.mean()
        else:
            d_l_old = 0
            dc_l_new = 0

        # Log the changes
        _, logp, _, v = actor_critic(obs, act)
        pi_l_new = -(logp*(k*adv+pos)).mean()
        v_l_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        logger.store(LossPi=pi_loss, LossV=v_l_old, KL=kl, Entropy=entropy, DeltaLossPi=(pi_l_new-pi_loss),
            DeltaLossV=(v_l_new-v_l_old), LossDC=d_l_old, DeltaLossDC=(dc_l_new-d_l_old))
        # logger.store(Adv=adv.reshape(-1).numpy().tolist(), Pos=pos.reshape(-1).numpy().tolist())

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t = 0

    for epoch in range(epochs):
        actor_critic.eval()
        disc.eval()
        for _ in range(local_episodes_per_epoch):
            c = context_dist.sample()
            c_onehot = F.one_hot(c, con_dim).squeeze().float()
            for _ in range(max_ep_len):
                concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
                a, _, logp_t, v_t = actor_critic(concat_obs)

                buffer.store(c, concat_obs.squeeze().detach().numpy(), a.detach().numpy(), r, v_t.item(), logp_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    _, _, log_p = disc(dc_diff, con)
                    buffer.end_episode(log_p.detach().numpy())
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [actor_critic, disc], None)

        # Update
        actor_critic.train()
        disc.train()

        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor')
    parser.add_argument('--con', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    valor(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid]*args.l),
        disc=Discriminator, dc_kwargs=dict(hidden_dims=args.hid),
        gamma=args.gamma, seed=args.seed, episodes_per_epoch=args.episodes, epochs=args.epochs, logger_kwargs=logger_kwargs, con_dim=args.con)

