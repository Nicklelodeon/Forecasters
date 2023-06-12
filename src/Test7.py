import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
from GeneticAlgoState import GeneticAlgoState
from GenAlgo import Genetic_Algo_No_Season
from GenAlgo import Genetic_Algo_Seasonal


""" algorithm_param = {'max_num_iteration': 100,\
                   'population_size':50,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

def objective(X):
    return -1 * Genetic_Algo_No_Season(X) #7 parameters """
""" def g(X):
    return -1 * genstate.GArun_no_season(X) #7 parameters """

#print(genstate.GArun_no_season([120,60,60,60,180,180,180]))

""" print(objective(np.array([122, 78, 182, 57, 152, 78, 182])))

varbound=np.array([[120, 180],[20,60], [120, 180], [20,60], [120, 180], [20,60], [120, 180]])




model2=ga(function=objective,\
        dimension=7,variable_type='int',\
        variable_boundaries=varbound, \
        algorithm_parameters=algorithm_param)

model2.run()  """

def objective2(X):
    return -1 * Genetic_Algo_Seasonal(X)

vbound = np.array([[20, 60],[120, 180]]*36)
model3 = ga(function=objective2,\
        dimension=72,\
        variable_type='int',\
        variable_boundaries=vbound, \
        algorithm_parameters=algorithm_param)
model3.run()
