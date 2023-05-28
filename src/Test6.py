from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand
from GenerateDemandMonthly import GenerateDemandMonthly

np.random.seed(1234)

synthetic = GenerateDemandMonthly()
synthetic.simulate_normal(1) #Simulating 1 year
demand_list = synthetic.simulate_poisson(1)

demandable_state = [-1, 0, 0, 1, 1]

state = State()
state.create_state(demandable_state)
state.set_demand_list(demand_list)
print(state.demand_list)
for i in range(len(state.demand_list)):
    print("-----------------------------")
    print("Demand " , state.demand_list[i])
    state.update_state(i)
    print(state.print_state(i))

print(state.root.calculate_profit())
print(state.rewards)
#78094