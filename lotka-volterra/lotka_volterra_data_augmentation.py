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

###########################
# Data generato functions #
###########################


def sample_squeeze(dist, p, dt, T):
    '''
    reshape param sample for use in ELBO
    '''
    dim2 = int(T / dt)
    sample = dist.sample([p, 1])
    return sample, tf.reshape(tf.tile(sample, [1, dim2]), [-1, 1])


def param_init(mean, std_init):
    '''
    init params of SDE as log normlas
    '''
    std = tf.log(tf.exp(std_init) - 1)
    param_mean = tf.Variable(mean)
    param_std = tf.nn.softplus(tf.Variable(std))
    param_dist = tfd.TransformedDistribution(distribution=tfd.Normal(
        loc=param_mean, scale=param_std), bijector=tfb.Exp())
    return param_dist, param_mean, param_std


def time_till(start_time, dt, obs_times):
    '''
    feature vector: generating the time until next obs
    '''
    time_store = np.array([])
    time_cat = np.concatenate((np.array([start_time]), obs_times), axis=0)
    time_diff = time_cat[1:] - time_cat[:-1]
    for i in range(len(time_diff)):
        time_store = np.append(time_store, np.linspace(
            time_diff[i], dt, int(np.round(time_diff[i] / dt, decimals=0))))
    return NP_DTYPE(time_store)[1:]


def obs_rep(obs, dt, start_time, obs_times):
    '''
    feature vector: time until the next observation
    '''
    obs_rep = np.array([])
    time_cat = np.concatenate((np.array([start_time]), obs_times), axis=0)
    time_diff = time_cat[1:] - time_cat[:-1]
    for i in range(len(time_diff)):
        obs_rep = np.append(obs_rep, np.repeat(
            obs[i], int(np.round(time_diff[i] / dt, decimals=0))))
    return NP_DTYPE(obs_rep)[1:]

#####################
# Data augmentation #
#####################


t0 = OBS_TIMES[0]
T = OBS_TIMES[-1]
x0 = OBS[:, 0]
x1_obs = OBS[0, 1:]
x2_obs = OBS[1, 1:]

tn_store = time_till(t0, DT, OBS_TIMES)

x1_store = obs_rep(x1_obs, DT, t0, OBS_TIMES[1:])
x2_store = obs_rep(x2_obs, DT, t0, OBS_TIMES[1:])

feature_init = np.tile(np.array([[[t0, OBS_TIMES[1] - t0, x1_store[0],
                                   x1_store[1], OBS[0, 1] - OBS[0, 0], OBS[1, 1] - OBS[1, 0]]]]), (P, 1, 1))

features = {'tn_store': tn_store, 'x1_store': x1_store, 'x2_store': x2_store, 'feature_init': feature_init}

# sample x0. not set to be learnable, but could swap in a variational
# approximate.
x_start = tf.expand_dims(tfd.MultivariateNormalDiag(
    loc=x0, scale_diag=[TAU] * 2).sample(P), axis=1)

obs = {'obs_init': x_start, 'obs': OBS, 'times': OBS_TIMES, 'tau': TAU}

#####################################
# Mean-field vartiational inference #
#####################################

# NOTE: This can very easily be swapped out for a more sophisticated
# variational approximate, e.g. IAF

no_input = feature_init.shape[2] + 2

c1_dist, c1_mean, c1_std = param_init(-.7, .1)
c1, c1_strech = sample_squeeze(c1_dist, P, DT, T)

c2_dist, c2_mean, c2_std = param_init(-6., .1)
c2, c2_strech = sample_squeeze(c2_dist, P, DT, T)

c3_dist, c3_mean, c3_std = param_init(-1.4, .1)
c3, c3_strech = sample_squeeze(c3_dist, P, DT, T)

params = {'c1': c1, 'c1_mean': c1_mean,
          'c1_std': c1_std, 'c1_strech': c1_strech,
          'c2': c2, 'c2_mean': c2_mean,
          'c2_std': c2_std, 'c2_strech': c2_strech,
          'c3': c3, 'c3_mean': c3_mean,
          'c3_std': c3_std, 'c3_strech': c3_strech}
