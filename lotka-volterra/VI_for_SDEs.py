import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# python data types
import numpy as np
import scipy.stats as stats
from datetime import datetime
# model-specific imports
from lotka_volterra_settings import *
from lotka_volterra_loss import ELBO
from network_utils import Weight, Bias

tfd = tf.contrib.distributions
tfb = tfd.bijectors

DTYPE = tf.float32
NP_DTYPE = np.float32

class Model():

    def __init__(self, num_layers, width, p, dt, sess):
        weights = {}

        for i in range(1, num_layers + 1):
            with tf.variable_scope('hidden_layer_%d' % i):
                if i == 1:
                    weights['w%i' % i] = Weight([1, no_input, width], DTYPE).tile(p)
                else:
                    weights['w%i' % i] = Weight([1, width, width], DTYPE).tile(p)
                weights['b%i' % i] = Bias([1, 1, width], DTYPE).tile(p)

        with tf.variable_scope('output_layer'):
            weights['w0'] = Weight([1, width, 5], DTYPE).tile(p)
            weights['b0'] = Bias([1, 1, 5], DTYPE).tile(p)

        self.sess = sess
        self.weights = weights
        self.num_layers = num_layers
        self.p = p
        self.dt = dt

    def build(self):
        print("Building graph...")
        # launching functions to create forward sims and calc the loss
        with tf.name_scope('diffusion_bridge'):
            paths, variational_mu, variational_sigma = self.diff_bridge(
                x_start, feature_init, self.weights, p, self.dt, T, params, tn_store, x1_store, x2_store, self.num_layers)

        with tf.name_scope('ELBO'):
            mean_loss = ELBO(obs, tau, paths, variational_mu, variational_sigma, params, priors, p, self.dt)
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

        self.mean_loss = mean_loss
        self.merged = tf.summary.merge_all()

    # diffusion bridge function: rolls out rnn cell across the time series. Is there a nice way to do this in tensorflow?
    def diff_bridge(self, x_start, feature_init, weights, p, dt, T, params, tn_store, x1_store, x2_store, num_layers):
        inp = tf.concat([x_start, feature_init], 2)
        pred_mu, pred_sigma = self.rnn_cell(inp, weights, p, params, num_layers)
        mu_store = tf.squeeze(pred_mu)
        sigma_store = tf.reshape(pred_sigma, [-1, 4])
        output = self.path_sampler(inp[:, 0, 0:2], pred_mu, pred_sigma, p, dt)
        path_store = tf.concat(
            [tf.reshape(inp[:, :, 0:2], [-1, 2, 1]), tf.reshape(output, [-1, 2, 1])], 2)

        for i in range(int(T / dt) - 1):
            x1_next_vec = tf.fill([p, 1, 1], x1_store[i])
            x2_next_vec = tf.fill([p, 1, 1], x2_store[i])

            inp = tf.concat([output, tf.tile([[[inp[0, 0, 2] + dt, tn_store[i], x1_store[i], x2_store[i]]]], [
                           p, 1, 1]), tf.concat([x1_next_vec, x2_next_vec], 2) - output], 2)
            pred_mu, pred_sigma = self.rnn_cell(
                inp, weights, p, params, num_layers)
            mu_store = tf.concat([mu_store, tf.squeeze(pred_mu)], 1)
            sigma_store = tf.concat([sigma_store, tf.reshape(pred_sigma, [-1, 4])], 1)
            output = self.path_sampler(inp[:, 0, 0:2], pred_mu, pred_sigma, p, dt)
            path_store = tf.concat([path_store, tf.reshape(output, [-1, 2, 1])], 2)

        sigma_store = tf.reshape(sigma_store, [-1, 2, 2])
        mu_store = tf.reshape(mu_store, [-1, 2])
        return path_store, mu_store, sigma_store

    # the rnn cell called by diff_bridge
    def rnn_cell(self, inp, weights, p, params, num_layers, eps_identity=1e-6):
        hidden_layer = tf.nn.relu(
            tf.add(tf.matmul(inp, weights['w1']), weights['b1']))

        for i in range(2, num_layers + 1):
            hidden_layer = tf.nn.relu(
                tf.add(tf.matmul(hidden_layer, weights['w%i' % i]), weights['b%i' % i]))

        output = tf.add(tf.matmul(hidden_layer, weights['w0']), weights['b0'])

        mu, sigma_11, sigma_21, sigma_22 = tf.split(output, [2, 1, 1, 1], 2)

        # reshaping sigma matrix to lower-triangular cholesky factor
        zeros = tf.zeros(tf.shape(sigma_11))
        sigma_11 = tf.nn.softplus(sigma_11)
        sigma_22 = tf.nn.softplus(sigma_22)
        sigma_chol = tf.concat([tf.concat([sigma_11, zeros], 2),
                                tf.concat([sigma_21, sigma_22], 2)], 1)
        sigma = tf.cholesky(tf.matmul(sigma_chol, tf.transpose(sigma_chol, perm=[0, 2, 1])) + eps_identity * tf.tile(
            tf.expand_dims(np.identity(2, dtype=np.float32), 0), [p, 1, 1]))

        return mu, sigma

    # functions to return p simulations of a diffison bridge
    def path_sampler(self, inp, mu_nn, sigma_nn, p, dt):
        out_dist = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalTriL(
            loc=inp + dt * tf.squeeze(mu_nn), scale_tril=tf.sqrt(dt) * sigma_nn), bijector=tfb.Softplus(event_ndims=1))
        out = tf.expand_dims(out_dist.sample(), 1)
        return out

    # train the model
    def train(self, niter, PATH):
        print("Running model...")
        writer = tf.summary.FileWriter(
            '%s/%s' % (PATH, datetime.now().strftime("%d:%m:%y-%H:%M:%S")), self.sess.graph)
        for i in range(niter):
            self.train_step.run()
            if i % 10 == 0:
                summary = self.sess.run(self.merged)
                writer.add_summary(summary, i)

    # functions to save and load models
    def save(self, SAVE_PATH):
        saver = tf.train.Saver()
        saver.save(self.sess, SAVE_PATH)
        print("Model Saved")

    def load(self, LOAD_PATH):
        saver = tf.train.Saver()
        saver.restore(self.sess, LOAD_PATH)
        print("Model Restored")
