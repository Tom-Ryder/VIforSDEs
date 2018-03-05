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
    a = tf.concat([params['c1_strech'] * x1 - params['c2_strech'] * x1 * x2,
                   params['c2_strech'] * x1 * x2 - params['c3_strech'] * x2], 1)
    return a


def beta(x1, x2, params):
    a = tf.expand_dims(tf.sqrt(params['c1_strech'] * x1 + params['c2_strech']* x1 * x2), 1)
    b = tf.expand_dims(- params['c2_strech'] * x1 * x2, 1) / a
    c = tf.sqrt(tf.expand_dims(params['c3_strech'] * x2 + params['c2_strech'] * x1 * x2, 1) - tf.square(b))
    zeros = tf.zeros(tf.shape(a))
    beta_chol = tf.concat([tf.concat([a, zeros], 2), tf.concat([b, c], 2)], 1)
    return beta_chol

# ELBO loss function

def ELBO(obs, tau, x_, mu_, sigma_, params, priors, p, dt):

    for i in range(1, int(len(obs)/2 + 1)):
        d1_mu = x_[:, :, int(obs['t%d' % i] / dt)]
        d1_std = [tau, tau]
        d1 = tfd.MultivariateNormalDiag(d1_mu, d1_std)
        inp = tf.tile(tf.expand_dims(
            obs['x%d' % i], 0), [p, 1])
        d1_temp = d1.log_prob(inp)
        if i == 1:
            d1_store = tf.expand_dims(d1_temp, 1)
        else:
            d1_store = tf.concat(
                [d1_store, tf.expand_dims(d1_temp, 1)], 1)
    d1_eval = tf.reduce_sum(d1_store, 1)

    x1_path = x_[:, 0, :]
    x2_path = x_[:, 1, :]

    x_path_diff = x_[:, :, 1:] - x_[:, :, :-1]
    x_diff = tf.concat([tf.reshape(x_path_diff[:, 0, :], [-1, 1]),
                        tf.reshape(x_path_diff[:, 1, :], [-1, 1])], 1)

    x_path_mean = tf.concat(
        [tf.reshape(x_[:, 0, :-1], [-1, 1]), tf.reshape(x_[:, 1, :-1], [-1, 1])], 1)
    x_path_eval = tf.concat(
        [tf.reshape(x_[:, 0, 1:], [-1, 1]), tf.reshape(x_[:, 1, 1:], [-1, 1])], 1)

    d2 = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalTriL(
        loc=x_path_mean + dt * mu_, scale_tril=tf.sqrt(dt) * sigma_), bijector=tfb.Softplus(event_ndims=1))

    d2_eval = d2.log_prob(x_path_eval)

    x1_head = tf.reshape(x1_path[:, :-1], [-1, 1])
    x2_head = tf.reshape(x2_path[:, :-1], [-1, 1])

    alpha_eval = alpha(x1_head, x2_head, params)
    beta_eval = beta(x1_head, x2_head, params)

    d3 = tfd.MultivariateNormalTriL(
        loc=dt * alpha_eval, scale_tril=tf.sqrt(dt) * beta_eval)
    d3_eval = d3.log_prob(x_diff)

    theta_cat = tf.log(tf.concat([params['c1'], params['c2'], params['c3']], 1))

    prior_dist = tfd.MultivariateNormalDiag(loc=[priors['c1_mean'], priors['c2_mean'], priors['c3_mean']],
                                            scale_diag=[priors['c1_std'], priors['c2_std'], priors['c3_std']])
    gen_dist = tfd.MultivariateNormalDiag(loc=[params['c1_mean'], params['c2_mean'], params['c3_mean']],
                                          scale_diag=[params['c1_std'], params['c2_std'], params['c3_std']])

    prior_loglik = prior_dist.log_prob(theta_cat)
    gen_loglik = gen_dist.log_prob(theta_cat)

    sum_eval = tf.reduce_sum(tf.reshape(d2_eval - d3_eval, [p, -1]), 1)
    loss = sum_eval - d1_eval + gen_loglik - prior_loglik
    mean_loss = tf.reduce_mean(loss, 0)

    return mean_loss
