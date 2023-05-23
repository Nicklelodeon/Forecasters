from BayesianOptimisation import BayesianOptimisation
from bayes_opt import BayesianOptimization

pbounds = {'s_DC1': (40, 70), 'S_DC1': (80, 120), 's_DC2': (40, 70), 'S_DC2': (80, 120), 's_r1': (40, 70), 'S_r1': (80, 120)}
optimizer = BayesianOptimization(
    f=BayesianOptimisation,
    pbounds=pbounds,
    random_state=0,
)
optimizer.maximize(
    init_points = 100,
    n_iter=1000
)
print(optimizer.max)