import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# weights, biases classes

DTYPE = tf.float32

class Weight:

    def __init__(self, shape):
        self.initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=DTYPE)

    def tile(self, p):
        return tf.tile(self.initial, [p, 1, 1])


class Bias:

    def __init__(self, shape):
        self.inital = tf.Variable(tf.fill(shape, 0.1), dtype=DTYPE)

    def tile(self, p):
        return tf.tile(self.inital, [p, 1, 1])
