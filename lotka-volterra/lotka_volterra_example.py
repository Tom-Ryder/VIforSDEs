import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# model-specific imports
from VI_for_SDEs import Model
from lotka_volterra_data import *

with tf.Session() as new_session:
    lotka_volterra = Model(num_layers=num_hidden_layers, width=hidden_layer_width, p = p, dt = dt, sess = new_session)
    lotka_volterra.build()
    new_session.run(tf.global_variables_initializer())
    # desired number of iterations. currently no implementation of a convergence criteria.
    lotka_volterra.train(25000, PATH_TO_TENSORBOARD_OUTPUT)
