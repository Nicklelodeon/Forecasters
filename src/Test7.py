import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State

state = State()
state.create_state([-1,0, 1, 1, 2, 2])



def f(X):
    return -1 * state.run(X)

varbound=np.array([[10,200]]*6)

model=ga(function=f,dimension=6,variable_type='int',variable_boundaries=varbound)

model.run()
