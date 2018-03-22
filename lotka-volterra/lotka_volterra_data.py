import numpy as np

# path to tensorboard output_layer
PATH_TO_TENSORBOARD_OUTPUT = "train/"

# input data: observations and times
inp = np.array([[  71.        ,   47.61225908,   80.53119269,   23.10087379,
         158.05238324],
       [  79.        ,  447.20971405,   50.26254069,  339.40432691,
          66.79611979]])

obs_times = np.arange(0,41,10)

# know observation error variance and discretisation
tau = 1.
dt = .1

# number of monte-carlo sims used for loss. must be greater than 1.
p = 50

# priors as lognormals.
priors = {'c1_mean': 0., 'c1_std': 3., 'c2_mean': 0.,
          'c2_std': 3., 'c3_mean': 0., 'c3_std': 3.}

# network settings: width and depth
num_hidden_layers = 4
hidden_layer_width = 50
