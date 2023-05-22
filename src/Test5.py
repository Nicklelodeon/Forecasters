from Demandable import Demandable
from Item import Item
from State import State
import numpy as np

np.random.seed(1234)

state = State()
state.create_state([-1,0, 1, 1, 2, 2])
print(state.changeable_network)


