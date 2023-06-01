# from BayesianOptimisation import BayesianOptimisation
# from bayes_opt import BayesianOptimization
# import random

# pbounds = {'s_DC1': random.sample(range(40, 91), 12), 'S_DC1': random.sample(range(40, 91), 12), 's_DC2': random.sample(range(40, 91), 12), 'S_DC2': random.sample(range(40, 91), 12), 's_r1': random.sample(range(40, 91), 12), 'S_r1': random.sample(range(40, 91), 12)}
# optimizer = BayesianOptimization(
#     f=BayesianOptimisation,
#     pbounds=pbounds,
#     random_state=0,
# )
# optimizer.maximize(
#     init_points = 100,
#     n_iter=1000
# )
# print(optimizer.max)

#code adapted from https://machinelearningmastery.com/what-is-bayesian-optimization/



# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from GenerateDemandMonthly import GenerateDemandMonthly
import random
import numpy as np 
from BayesianState import BayesianState
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern

 
# objective function
def objective(lst):
    n = len(lst) // 6
    s_DC1 = lst[:n]
    S_DC1 = lst[n:2*n]
    s_DC2 = lst[2*n:3*n]
    S_DC2 = lst[3*n:4*n]
    s_r1 = lst[4*n:5*n] 
    S_r1 = lst[5*n:]
    demand = GenerateDemandMonthly()
    state = BayesianState()
    state.create_state([-1 ,0, 1, 1, 2, 2])
    total_sum = 0
    for z in range(30):
        state.set_demand_list(demand.simulate_normal(1))
        for i in range(12):
            if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
                    return -100000
            state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
            state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
            state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
            state.update_state(i)
        total_sum += state.rewards
        state.reset()
    return total_sum / 30
 
# surrogate or approximation for the objective function
def surrogate(model, X):
 # catch any warning generated when making a prediction
 with catch_warnings():
 # ignore generated warnings
    simplefilter("ignore")
 return model.predict(X.reshape(1, -1), return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
 # calculate the best surrogate score found so far
    best = 0
    for i in range(len(X)):
        yhat, _ = surrogate(model, X[i])
        best = max(yhat, best)
 # calculate mean and stdev via surrogate function
    best_mu = 0
    best_std = 0
    for i in range(len(Xsamples)):
        mu, std = surrogate(model, Xsamples[i])
        mu = max(mu, best_mu)
        std = max(std, best_std)
    #  mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((best_mu - best) / (best_std+1E-9))
    return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):
 # random search, generate random samples
    Xsamples = []

    for x in range(100):
        z = []
        for y in range(3):
            z.extend(random.choices(range(20, 50), k=12))
            z.extend(random.choices(range(60, 90), k=12))
        Xsamples.append(z)
    Xsamples = asarray(Xsamples)
    #  Xsamples = Xsamples.reshape(1200, 6)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix]
 
# # plot real observations vs surrogate function
# def plot(X, y, model):
#  # scatter plot of inputs and real objective function
#  pyplot.scatter(X, y)
#  # line plot of surrogate function across domain
#  Xsamples = asarray(arange(0, 1, 0.001))
#  Xsamples = Xsamples.reshape(len(Xsamples), 1)
#  ysamples, _ = surrogate(model, Xsamples)
#  pyplot.plot(Xsamples, ysamples)
#  # show the plot
#  pyplot.show()
 
# sample the domain sparsely with noise
X = []
for x in range(100):
    z = []
    for y in range(3):
        z.extend(random.choices(range(20, 50), k=12))
        z.extend(random.choices(range(60, 90), k=12))
    X.append((z))
X = asarray(X)
y = asarray([objective(lst) for lst in X])
# reshape into rows and cols
# X = X.reshape(100, 72)
y = y.reshape(-1, 1)
# define the model
kernel = Matern()
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# fit the model
for i in range(len(y)):
    model.fit(X[i].reshape(1, -1), y[i])
# plot before hand
plot(X, y, model)
# perform the optimization process
for i in range(100):
 # select the next point to sample
 x = opt_acquisition(X, y, model)
 # sample the point
 actual = objective(x)
 # summarize the finding
 print(x[0])

 est, _ = surrogate(model, x.reshape(1, -1))
 print('>x=%s, f()=%3f, actual=%.3f' % (str(x), est[0], actual))
 # add the data to the dataset
 X = vstack((X, [x]))
 y = vstack((y, [actual]))
 # update the model
 model.fit(X, y)
 
# plot all samples and the final surrogate function
# plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%s, y=%.3f' % (str(X[ix]), y[ix]))

# Best Result: x=[25 48 33 30 40 33 49 37 44 41 30 32 60 63 65 63 63 87 69 78 88 61 71 74
#  44 37 32 47 22 34 28 38 46 26 20 32 66 88 63 87 67 84 83 73 76 63 84 78
#  23 41 45 44 35 44 28 29 42 26 21 23 83 77 73 79 79 80 80 73 79 74 78 62], y=16749.100

# Best Result: x=[27 24 30 38 21 36 48 47 40 40 34 23 72 87 70 67 67 76 69 62 62 77 60 72
#  44 41 37 29 48 46 44 31 25 47 45 23 76 67 64 68 88 84 84 64 73 81 68 64
#  48 49 34 47 46 24 30 38 35 49 37 27 86 80 80 82 67 87 87 76 66 84 75 64], y=16550.953