import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
import pandas as pd

algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':100,\
                   'mutation_probability':0.75,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':300}

df = pd.read_csv("./TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
varbound=np.array([[round(mean * 2), round(mean * 10)]]*6)

def objective_no_season(X):
    """Objective function

    Args:
        X (Array): _description_

    Returns:
        _type_: _description_
    """
    s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r2 = X
    return -1 * state.run(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r2) #6 parameters

model=ga(function=objective_no_season,\
        dimension=6,variable_type='int',\
        variable_boundaries=varbound, \
        algorithm_parameters=algorithm_param)

model.run() 