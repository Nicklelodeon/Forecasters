import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
from GeneticAlgoState import GeneticAlgoState
from scipy.optimize import minimize

genstate = GeneticAlgoState()
genstate.create_state([-1,0, 1, 1, 2, 2])

def g(X):
    return -genstate.GArun_no_season(X) #7 parameters


varbound= [[120, 200],[20, 80], [20,80], [20,80], [120, 200], [120, 200], [120, 200]]

varbound = list(map(lambda x: (x[0],x[1]),varbound))
print(varbound)

x0 = np.array([160, 50, 50, 50, 160, 160, 160])

result = minimize(g, x0, method = 'SLSQP', bounds=varbound)
#print(result)
