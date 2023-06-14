from skopt import gp_minimize
import numpy as np
# np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from BayesianOptimisation import BayesianOptimisation
import pandas as pd 

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

res = gp_minimize(BayesianOptimisation,                  # the function to minimize
                  [(round(mean * 2), round(mean * 10), round(mean * 2), round(mean * 10), round(mean * 2), round(mean * 10), round(mean * 2), round(mean * 10), round(mean * 2), round(mean * 10), round(mean * 2), round(mean * 10), round(mean * 2), round(mean * 10))],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

from skopt.plots import plot_convergence
plot_convergence(res)


