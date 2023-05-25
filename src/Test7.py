import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
from GeneticAlgoState import GeneticAlgoState


genstate = GeneticAlgoState()
genstate.create_state([-1,0, 1, 1, 2, 2])


print(-1 * genstate.GArun([79, 88, 51, 53, 41, 43]))

def f(X):
    return -1 * genstate.GArun(X)

varbound=np.array([[40,90]]*6)

algorithm_param = {'max_num_iteration': 500,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

#model=ga(function=f,dimension=6,variable_type='int',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

#model.run()