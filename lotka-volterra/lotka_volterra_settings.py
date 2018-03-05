import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# python data types
import numpy as np
# model-specific data
from lotka_volterra_data import *

tfd = tf.contrib.distributions
tfb = tfd.bijectors

DTYPE = tf.float32
NP_DTYPE = np.float32

# functions to init variables and augment data

def sample_squeeze(dist, p, dt, T):
    dim2 = int(T / dt)
    sample = dist.sample([p, 1])
    return sample, tf.reshape(tf.tile(sample, [1, dim2]), [-1, 1])


def param_init(mean, std_init):
    std = tf.log(tf.exp(std_init) - 1)
    param_mean = tf.Variable(mean)
    param_std = tf.nn.softplus(tf.Variable(std))
    ds = tf.contrib.distributions
    param_dist = tfd.TransformedDistribution(distribution=tfd.Normal(
        loc=param_mean, scale=param_std), bijector=tfb.Exp())
    return param_dist, param_mean, param_std


def start_dist(mean, std):
    mean = tf.cast(mean, tf.float64)
    std = tf.cast(std, tf.float64)
    start_dist = tfd.Normal(loc=mean, scale=std)
    return start_dist


def start_sample(x1_dist, x2_dist, p):
    out_1 = x1_dist.sample([p, 1, 1])
    out_2 = x2_dist.sample([p, 1, 1])
    out = tf.concat([out_1, out_2], 2)
    return tf.cast(out, DTYPE)


def obs_gen(obs_list, time_list):
    obs = {}
    obs_list = inp
    time_list = obs_times
    i = 1
    for i in range(1, obs_list.shape[1]):
        obs['x%d' % (i)] = NP_DTYPE(obs_list[:, i])
        obs['t%d' % (i)] = time_list[i]
    return obs, time_list[0], time_list[obs_list.shape[1] - 1], obs_list[0, 0], obs_list[1, 0]


def time_till(start_time, dt, obs_times):
    time_store = np.array([])
    time_cat = np.concatenate((np.array([start_time]), obs_times), axis=0)
    time_diff = time_cat[1:] - time_cat[:-1]
    for i in range(len(time_diff)):
        time_store = np.append(time_store, np.linspace(
            time_diff[i], dt, int(np.round(time_diff[i] / dt, decimals=0))))
    return NP_DTYPE(time_store)[1:]


def obs_rep(obs, dt, start_time, obs_times):
    obs_rep = np.array([])
    time_cat = np.concatenate((np.array([start_time]), obs_times), axis=0)
    time_diff = time_cat[1:] - time_cat[:-1]
    for i in range(len(time_diff)):
        obs_rep = np.append(obs_rep, np.repeat(
            obs[i], int(np.round(time_diff[i] / dt, decimals=0))))
    return NP_DTYPE(obs_rep)[1:]

# ---


obs, start_time, T, x1, x2 = obs_gen(inp, obs_times)

x1_obs = inp[0, 1:]
x2_obs = inp[1, 1:]

tn_store = time_till(start_time, dt, obs_times)

x1_store = obs_rep(x1_obs, dt, start_time, obs_times[1:])
x2_store = obs_rep(x2_obs, dt, start_time, obs_times[1:])

feature_init = np.tile(np.array([[[start_time, obs['t1'] - start_time, x1_store[0],
                                   x1_store[1], inp[0, 1] - inp[0, 0], inp[1, 1] - inp[1, 0]]]]), (p, 1, 1))

# params

no_input = 8

c1_dist, c1_mean, c1_std = param_init(-.7, .1)
c1, c1_strech = sample_squeeze(c1_dist, p, dt, T)

c2_dist, c2_mean, c2_std = param_init(-6., .1)
c2, c2_strech = sample_squeeze(c2_dist, p, dt, T)

c3_dist, c3_mean, c3_std = param_init(-1.4, .1)
c3, c3_strech = sample_squeeze(c3_dist, p, dt, T)

params = {'c1': c1, 'c1_mean': c1_mean,
          'c1_std': c1_std, 'c1_strech': c1_strech,
          'c2': c2, 'c2_mean': c2_mean,
          'c2_std': c2_std, 'c2_strech': c2_strech,
          'c3': c3, 'c3_mean': c3_mean,
          'c3_std': c3_std, 'c3_strech': c3_strech}

# start samples

x1_start_dist = start_dist(x1, tau)
x2_start_dist = start_dist(x2, tau)
x_start = start_sample(x1_start_dist, x2_start_dist, p)
