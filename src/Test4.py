from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from Retailer import Retailer

np.random.seed(1234)

demandable_state = [-1, 0, 0, 1, 1]

state = State()
state.create_state(demandable_state)
state.print_network()
print(state.demand_list)

for i in range(6):
    state.update_state(25, i)
    print(state.print_state(i))