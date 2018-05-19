import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# python data types
import numpy as np
import scipy.stats as stats
from datetime import datetime
# model-specific imports
from lotka_volterra_data_augmentation import *
from lotka_volterra_loss import ELBO
from network_utils import Weight, Bias

tfd = tf.contrib.distributions
tfb = tfd.bijectors

DTYPE = tf.float32
NP_DTYPE = np.float32


class Model():

    def __init__(self, network_params, p, dt, T, obs, params, priors, features):
        '''
        :params num_layers: number of hidden layers in NN
        :params width: width of the hidden layers
        :params p: number of particles for monte-carlo esitmate
        :params dt: discretisation
        :params sess: passing current tensorflow sess to Model train()
        '''
        weights = {}

        for i in range(1, network_params['num_hidden_layers'] + 1):
            with tf.variable_scope('hidden_layer_%d' % i):
                if i == 1:
                    weights['w%i' % i] = Weight(
                        [1, no_input, network_params['hidden_layer_width']], DTYPE).tile(p)
                else:
                    weights['w%i' % i] = Weight(
                        [1, network_params['hidden_layer_width'], network_params['hidden_layer_width']], DTYPE).tile(p)
                weights['b%i' % i] = Bias(
                    [1, 1, network_params['hidden_layer_width']], DTYPE).tile(p)

        with tf.variable_scope('output_layer'):
            weights['w0'] = Weight(
                [1, network_params['hidden_layer_width'], 5], DTYPE).tile(p)
            weights['b0'] = Bias([1, 1, 5], DTYPE).tile(p)

        self.weights = weights
        self.network_params = network_params
        self.obs = obs
        self.params = params
        self.priors = priors
        self.features = features
        self.p = p
        self.dt = dt
        self.T = T

        # building computational graph
        self._build()

    def _build(self):
        '''
        buidling model graph
        '''
        print("Building graph...")
        # launching functions to create forward sims and calc the loss
        with tf.name_scope('diffusion_bridge'):
            paths, variational_mu, variational_sigma = self._diff_bridge()

        with tf.name_scope('ELBO'):
            mean_loss = ELBO(self.obs, paths, variational_mu,
                             variational_sigma, self.params, self.priors, self.p, self.dt)
            tf.summary.scalar('mean_loss', mean_loss)

        # specifying optimizer and gradient clipping for backprop
        with tf.name_scope('optimize'):
            opt = tf.train.AdamOptimizer(1e-3)
            gradients, variables = zip(
                *opt.compute_gradients(mean_loss))
            global_norm = tf.global_norm(gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, 4e3)
            self.train_step = opt.apply_gradients(
                zip(gradients, variables))
            tf.summary.scalar('global_grad_norm', global_norm)

        # mean-field approx params to tensorboard
        with tf.name_scope('variables'):
            with tf.name_scope('theta1'):
                tf.summary.scalar('theta1_mean', c1_mean)
                tf.summary.scalar('theta1_std', c1_std)
            with tf.name_scope('theta2'):
                tf.summary.scalar('theta2_mean', c2_mean)
                tf.summary.scalar('theta2_std', c2_std)
            with tf.name_scope('theta3'):
                tf.summary.scalar('theta3_mean', c3_mean)
                tf.summary.scalar('theta3_std', c3_std)

        self.merged = tf.summary.merge_all()

    def _diff_bridge(self):
        '''
        rolls out rnn cell across the time series
        '''
        inp = tf.concat([self.obs['obs_init'], self.features['feature_init']], 2)
        pred_mu, pred_sigma = self._rnn_cell(inp)
        mu_store = tf.squeeze(pred_mu)
        sigma_store = tf.reshape(pred_sigma, [-1, 4])
        output = self._path_sampler(inp[:, 0, 0:2], pred_mu, pred_sigma)
        path_store = tf.concat(
            [tf.reshape(inp[:, :, 0:2], [-1, 2, 1]), tf.reshape(output, [-1, 2, 1])], 2)

        for i in range(int(self.T / self.dt) - 1):
            x1_next_vec = tf.fill([self.p, 1, 1], self.features['x1_store'][i])
            x2_next_vec = tf.fill([self.p, 1, 1], self.features['x2_store'][i])

            inp = tf.concat([output, tf.tile([[[inp[0, 0, 2] + self.dt, self.features['tn_store'][i], self.features['x1_store'][i], self.features['x2_store'][i]]]], [
                self.p, 1, 1]), tf.concat([x1_next_vec, x2_next_vec], 2) - output], 2)
            pred_mu, pred_sigma = self._rnn_cell(inp)
            mu_store = tf.concat([mu_store, tf.squeeze(pred_mu)], 1)
            sigma_store = tf.concat(
                [sigma_store, tf.reshape(pred_sigma, [-1, 4])], 1)
            output = self._path_sampler(inp[:, 0, 0:2], pred_mu, pred_sigma)
            path_store = tf.concat(
                [path_store, tf.reshape(output, [-1, 2, 1])], 2)

        sigma_store = tf.reshape(sigma_store, [-1, 2, 2])
        mu_store = tf.reshape(mu_store, [-1, 2])
        return path_store, mu_store, sigma_store

    # the rnn cell called by diff_bridge
    def _rnn_cell(self, inp, eps_identity=1e-3):
        '''
        rnn cell for supplying Gaussian state transitions
        '''
        hidden_layer = tf.nn.relu(
            tf.add(tf.matmul(inp, self.weights['w1']), self.weights['b1']))

        for i in range(2, self.network_params['num_hidden_layers'] + 1):
            hidden_layer = tf.nn.relu(
                tf.add(tf.matmul(hidden_layer, self.weights['w%i' % i]), self.weights['b%i' % i]))

        output = tf.add(
            tf.matmul(hidden_layer, self.weights['w0']), self.weights['b0'])

        mu, sigma_11, sigma_21, sigma_22 = tf.split(output, [2, 1, 1, 1], 2)

        # reshaping sigma matrix to lower-triangular cholesky factor
        zeros = tf.zeros(tf.shape(sigma_11))
        sigma_11 = tf.nn.softplus(sigma_11)
        sigma_22 = tf.nn.softplus(sigma_22)
        sigma_chol = tf.concat([tf.concat([sigma_11, zeros], 2),
                                tf.concat([sigma_21, sigma_22], 2)], 1)
        sigma = tf.cholesky(tf.matmul(sigma_chol, tf.transpose(sigma_chol, perm=[0, 2, 1])) + eps_identity * tf.tile(
            tf.expand_dims(np.identity(2, dtype=np.float32), 0), [self.p, 1, 1]))

        return mu, sigma

    # functions to return p simulations of a diffison bridge
    def _path_sampler(self, inp, mu_nn, sigma_nn):
        '''
        sample new state using learned Gaussian state transitions
        :param inp: current state of system
        :param mu_nn: drift vector from RNN
        :param sigma_nn: diffusion matrix from RNN as cholesky factor
        '''
        out_dist = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalTriL(
            loc=inp + self.dt * tf.squeeze(mu_nn), scale_tril=tf.sqrt(self.dt) * sigma_nn), bijector=tfb.Softplus(event_ndims=1))
        out = tf.expand_dims(out_dist.sample(), 1)
        return out

    # train the model
    def train(self, niter, path):
        '''
        trains model
        :params niter: number of iterations
        :params PATH: path to tensorboard output
        '''
        print("Training model...")
        writer = tf.summary.FileWriter(
            '%s/%s' % (path, datetime.now().strftime("%d:%m:%y-%H:%M:%S")), sess.graph)
        for i in range(niter):
            self.train_step.run()
            if i % 10 == 0:
                summary = sess.run(self.merged)
                writer.add_summary(summary, i)

    def save(self, path):
        '''
        save model
        '''
        saver = tf.train.Saver()
        saver.save(sess, path)
        print("Model Saved")

    def load(self, path):
        '''
        load model
        '''
        saver = tf.train.Saver()
        saver.restore(sess, path)
        print("Model Restored")


if __name__ == "__main__":
    with tf.Session() as sess:
        lotka_volterra = Model(network_params=NETWORK_PARAMS, p=P,
                               dt=DT, T=T, obs=obs, params=params, priors=PRIORS, features=features)
        sess.run(tf.global_variables_initializer())
        # desired number of iterations. currently no implementation of a
        # convergence criteria.
        lotka_volterra.train(25000, PATH_TO_TENSORBOARD_OUTPUT)
