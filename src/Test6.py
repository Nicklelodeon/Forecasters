from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand

np.random.seed(1234)

synthetic = GenerateDemand()
synthetic.simulate(1) #Simulating 1 year
demand_list = synthetic.get_demand()

demandable_state = [-1, 0, 0, 1, 1]

state = State()
state.create_state(demandable_state)
state.print_network()
print(demand_list)

""" for i in range(len(demand_list)):
    print("Demand " , demand_list[i])
    state.update_state(demand_list[i], i)
    print(state.print_state(i)) """