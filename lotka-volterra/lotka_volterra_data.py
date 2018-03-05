import numpy as np

# path to tensorboard output_layer
PATH_TO_TENSORBOARD_OUTPUT = "train/"

# input data: observations and times
inp = np.array([[71.        ,  93.79751636],
       [  79.        ,  69.22939593]])

obs_times = np.arange(0,2)

# know observation error variance and discretisation
tau = 1.
dt = .1

# number of monte-carlo sims used for loss
p = 50

# priors as lognormals.
priors = {'c1_mean': 0., 'c1_std': 3., 'c2_mean': 0.,
          'c2_std': 3., 'c3_mean': 0., 'c3_std': 3.}
