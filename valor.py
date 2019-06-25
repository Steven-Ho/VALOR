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

def valor(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), disc=Discriminator, dc_kwargs=dict(), seed=0, episodes_per_epoch=40, epochs=50, gamma=0.99,
        pi_lr=3e-4, vf_lr=1e-3, dc_lr=5e-4, train_v_iters=80, lam=0.97, max_ep_len=1000, logger_kwargs=dict(), con_dim=5, save_freq=10):

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
    actor_critic = actor_critic(input_dim=obs_dim[0], **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    print(obs_dim)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = Buffer(con_dim, obs_dim[0], act_dim, local_episodes_per_epoch, max_ep_len)

    # Count variables
    var_counts = tuple(count_vars(module) for module in
        [actor_critic.policy, actor_critic.value_function])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)    

    # Optimizers
    train_pi = torch.optim.Adam(actor_critic.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(actor_critic.value_f.parameters(), lr=vf_lr)
    train_dc = torch.optim.Adam(disc.policy.parameters(), lr=dc_lr)

    # Parameters Sync
    sync_all_params(actor_critic.parameters())
    sync_all_params(disc.parameters())

    def update():
        obs, act, adv, ret, logp_old, s_diff = [torch.Tensor(x) for x in buffer.retrive_all()]
        pass

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    context_dist = Categorical(logits=np.ones(con_dim))
    total_t = 0

    for epoch in range(epochs):
        actor_critic.eval()
        disc.eval()
        for t in range(local_episodes_per_epoch):
            c = context_dist.sample()
            c = F.one_hot(c, con_dim).squeeze()
            for s in range(max_ep_len):
                a, _, logp_t, v_t = actor_critic(torch.Tensor(o.reshape(1, -1)))

                buffer.store(c, o, a.detach().numpy(), r, v_t.item(), logp_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a,detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [actor_critic, disc], None)

        # Update
        actor_critic.train()
        disc.train()

        update()

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
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
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='valor')
    parser.add_argument('--con', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    valor(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid]*args.l),
        disc=Discriminator, dc_kwargs=dict(hidden_dims=args.hid),
        gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs, con_dim=args.con)

