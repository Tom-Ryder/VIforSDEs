import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# weights, biases classes

class Weight:

    def __init__(self, shape, dtype):
        self.initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=dtype)

    def tile(self, p):
        return tf.tile(self.initial, [p, 1, 1])


class Bias:

    def __init__(self, shape, dtype):
        self.inital = tf.Variable(tf.fill(shape, 0.1), dtype=dtype)

    def tile(self, p):
        return tf.tile(self.inital, [p, 1, 1])
