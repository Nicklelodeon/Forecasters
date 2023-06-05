import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
from GeneticAlgoState import GeneticAlgoState
import time


genstate = GeneticAlgoState()
genstate.create_state([-1,0, 1, 1, 2, 2])

""" start_time = time.time()

for i in range(250):
    -1 * genstate.GArun2([40,90]* 12 * 3)

end_time = time.time()

run = (end_time - start_time)/250 

print("On average, Program Ran for:", run)"""
#0.04909294605255127
def f(X):
    return -1 * genstate.GArun2(X)

varbound=np.array([[40,90]]*72)

algorithm_param = {'max_num_iteration': 100,\
                   'population_size':500,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=f,dimension=72,variable_type='int',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()