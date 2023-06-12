from Demandable import Demandable
from Item import Item
from State import State
import numpy as np

np.random.seed(1234)

state = State()
state.create_state([-1,0, 1, 1, 2, 2])
state.create_array(40, 60, 80, 100)
print(state.changeable_network)
print(state.s_S_list)

for i in range(4):
    state.update_state(i)
    print(state.print_state(i))
    print("demand: " + str(state.demand_list[i]))
    print("cost function: " + str(state.rewards[i]))

print(state.total_sum())
