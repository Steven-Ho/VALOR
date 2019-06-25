import numpy as np
import torch
import torch.nn.functional as F
import gym
import time
import scipy.signal
from utils.logx import EpochLogger
from utils.mpi_torch import average_gradients, sync_all_params
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs

class Buffer(object):
    def __init__(self, con_dim, obs_dim, act_dim, batch_size, ep_len, gamma=0.99, lam=0.95, N=11):
        self.max_batch = batch_size
        self.max_s = batch_size * ep_len
        self.obs = np.zeros((self.max_s, obs_dim))
        self.act = np.zeros((self.max_s, act_dim))
        self.con = np.zeros((self.max_batch, con_dim))
        self.rew = np.zeros(self.max_s)
        self.ret = np.zeros(self.max_s)
        self.adv = np.zeros(self.max_s)
        self.lgt = np.zeros(self.max_s)
        self.val = np.zeros(self.max_s)
        self.end = np.zeros(batch_size + 1) # The first will always be 0
        self.ptr = 0
        self.eps = 0

        self.N = 11

        self.dcbuf = np.zeros((self.max_batch, self.N-1, obs_dim))

        self.gamma = gamma
        self.lam = lam

    def store(self, con, obs, act, rew, val, lgt):
        assert self.ptr < self.max_s
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.con[self.eps] = con
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.lgt[self.ptr] = lgt
        self.ptr += 1

    def end_episode(self, last_val=0):
        ep_slice = slice(self.end[self.eps], self.ptr)
        rewards = np.append(self.rew[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]

        # Store differences into a specific memory
        # TODO: convert this into vector operation
        start = self.end[self.eps]
        ep_l = self.ptr - start - 1
        for i in range(self.N):
            prev = int(i*ep_l/self.N)
            succ = int((i+1)*ep_l/self.N)
            self.dcbuf[self.eps, i] = self.obs[start + succ] - self.obs[start + prev]

        self.eps += 1
        self.end[self.eps] = self.ptr

    def retrive_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        adv_mean, adv_std = mpi_statistics_scalar(self.adv[occup_slice])
        self.adv[occup_slice] = (self.adv[occup_slice] - adv_mean) / adv_std
        return [self.obs[occup_slice], self.act[occup_slice], self.adv[occup_slice],
            self.ret[occup_slice], self.lgt[occup_slice], self.con, self.dcbuf]