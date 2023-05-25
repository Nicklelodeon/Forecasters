from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand
from GenerateDemandMonthly import GenerateDemandMonthly

np.random.seed(1234)

synthetic = GenerateDemandMonthly()
synthetic.simulate_normal(1) #Simulating 1 year
demand_list = synthetic.get_demand()

demandable_state = [-1, 0, 0, 1, 1]

state = State()
state.create_state(demandable_state)
print(state.demand_list)

for i in range(len(state.demand_list)):
    print("-----------------------------")
    print("Demand " , state.demand_list[i])
    state.update_state(i)
    print(state.print_state(i))