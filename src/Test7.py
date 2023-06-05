import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
from GeneticAlgoState import GeneticAlgoState

genstate = GeneticAlgoState()
genstate.create_state([-1,0, 1, 1, 2, 2])

def g(X):
    return -1 * genstate.GArun_no_season(X) #7 parameters

#print(genstate.GArun_no_season([120,60,60,60,180,180,180]))

varbound=np.array([[120, 180],[20,60], [20,60], [20,60], [120, 180], [120, 180], [120, 180]])

algorithm_param = {'max_num_iteration': 100,\
                   'population_size':50,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


model2=ga(function=g,dimension=7,variable_type='int',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model2.run()

