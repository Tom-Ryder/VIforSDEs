import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# python data types
import numpy as np

tfd = tf.contrib.distributions
tfb = tfd.bijectors

DTYPE = tf.float32
NP_DTYPE = np.float32

# functions for mu (alpha) and sigma (beta)
def alpha(x1, x2, params):
    '''
    returns drift vector for approx p(x)
    '''
    a = tf.concat([params['c1_strech'] * x1 - params['c2_strech'] * x1 * x2,
                   params['c2_strech'] * x1 * x2 - params['c3_strech'] * x2], 1)
    return a


def beta(x1, x2, params):
    '''
    returns diffusion matrix for approx p(x)
    '''
    a = tf.expand_dims(tf.sqrt(params['c1_strech'] * x1 + params['c2_strech']* x1 * x2), 1)
    b = tf.expand_dims(- params['c2_strech'] * x1 * x2, 1) / a
    c = tf.sqrt(tf.expand_dims(params['c3_strech'] * x2 + params['c2_strech'] * x1 * x2, 1) - tf.square(b))
    zeros = tf.zeros(tf.shape(a))
    beta_chol = tf.concat([tf.concat([a, zeros], 2), tf.concat([b, c], 2)], 1)
    return beta_chol

# ELBO loss function
def ELBO(obs, vi_paths, vi_mu, vi_sigma, params, priors, p, dt):
    '''
    calculate ELBO under SDE model
    :param obs: observations of SDE
    :param vi_paths: diffusion paths produced by generative VI approx
    :param vi_mu: drift vectors produced by generative VI approx
    :param vi_sigma: diffusion matrices produced by generative VI approx
    :param params: current params of model
    :param p: number of samples used for monte-carlo estimate
    :param dt: discretization used
    '''

    time_index = np.int32(obs['times'] / dt)

    for i in range(len(time_index)):
        obs_lik = tfd.MultivariateNormalDiag(vi_paths[:, :, time_index[i]], scale_identity_multiplier = tf.sqrt([obs['tau']]))
        obs_loglik = obs_lik.log_prob(tf.tile(tf.expand_dims(obs['obs'][:, i], 0), [p, 1]))
        if i == 0:
            obs_logprob_store = tf.expand_dims(obs_loglik, 1)
        else:
            obs_logprob_store = tf.concat(
                [obs_logprob_store, tf.expand_dims(obs_loglik, 1)], 1)
    obs_logprob = tf.reduce_sum(obs_logprob_store, 1)

    x1_path = vi_paths[:, 0, :]
    x2_path = vi_paths[:, 1, :]

    x_path_diff = vi_paths[:, :, 1:] - vi_paths[:, :, :-1]
    x_diff = tf.concat([tf.reshape(x_path_diff[:, 0, :], [-1, 1]),
                        tf.reshape(x_path_diff[:, 1, :], [-1, 1])], 1)

    x_path_mean = tf.concat(
        [tf.reshape(vi_paths[:, 0, :-1], [-1, 1]), tf.reshape(vi_paths[:, 1, :-1], [-1, 1])], 1)
    x_path_eval = tf.concat(
        [tf.reshape(vi_paths[:, 0, 1:], [-1, 1]), tf.reshape(vi_paths[:, 1, 1:], [-1, 1])], 1)

    gen_dist = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalTriL(
        loc=x_path_mean + dt * vi_mu, scale_tril=tf.sqrt(dt) * vi_sigma), bijector=tfb.Softplus(event_ndims=1))

    gen_logprob = gen_dist.log_prob(x_path_eval)

    x1_head = tf.reshape(x1_path[:, :-1], [-1, 1])
    x2_head = tf.reshape(x2_path[:, :-1], [-1, 1])

    alpha_eval = alpha(x1_head, x2_head, params)
    beta_eval = beta(x1_head, x2_head, params)

    sde_dist = tfd.MultivariateNormalTriL(
        loc=dt * alpha_eval, scale_tril=tf.sqrt(dt) * beta_eval)
    sde_logprob = sde_dist.log_prob(x_diff)

    theta_cat = tf.log(tf.concat([params['c1'], params['c2'], params['c3']], 1))

    prior_dist = tfd.MultivariateNormalDiag(loc=[priors['c1_mean'], priors['c2_mean'], priors['c3_mean']],
                                            scale_diag=[priors['c1_std'], priors['c2_std'], priors['c3_std']])
    gen_dist = tfd.MultivariateNormalDiag(loc=[params['c1_mean'], params['c2_mean'], params['c3_mean']],
                                          scale_diag=[params['c1_std'], params['c2_std'], params['c3_std']])

    prior_loglik = prior_dist.log_prob(theta_cat)
    gen_loglik = gen_dist.log_prob(theta_cat)

    sum_eval = tf.reduce_sum(tf.reshape(gen_logprob - sde_logprob, [p, -1]), 1)
    loss = sum_eval - obs_logprob + gen_loglik - prior_loglik
    mean_loss = tf.reduce_mean(loss, 0)

    return mean_loss
