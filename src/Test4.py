from Demandable import Demandable
from Item import Item
import numpy as np
from State import State

np.random.seed(1234)

demandable_state = [-1, 0, 0, 1, 1]

state = State()
state.create_state(demandable_state)
state.print_network()
#print("HERE AT LINE 11", state)

for i in range(100):
    state.update_state(1000, i)
    print(state.print_state(i))