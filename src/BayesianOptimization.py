#adapted from https://github.com/bayesian-optimization/BayesianOptimization
from bayes_opt import BayesianOptimization
import pandas as pd
from State import State

df = pd.read_csv("./TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()
state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

pbounds = {'s_DC1': (round(mean * 2), round(mean * 10)), 'S_DC1': (round(mean * 2), round(mean * 10)), 's_DC2': (round(mean * 2), round(mean * 10)), 'S_DC2': (round(mean * 2), round(mean * 10)), 's_r1':(round(mean * 2), round(mean * 10)), 'S_r1': (round(mean * 2), round(mean * 10))}
optimizer = BayesianOptimization(
    f=state.run,
    pbounds=pbounds,
    random_state=0,
    allow_duplicate_points=True
)
optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

optimizer.maximize(
    init_points = 100,
    n_iter=1500
)

print(optimizer.max)