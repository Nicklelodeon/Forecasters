from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand
from GenerateDemandMonthly import GenerateDemandMonthly

np.random.seed(1234)
demandable_state = [-1,0, 0, 1, 1]
demand = GenerateDemandMonthly()

state = State()
state.create_state(demandable_state)

state.set_demand_list(demand.simulate_normal_no_season())
print(state.root.s)
state.show_network()
print(state.demand_list)


for i in range(10):
    print("-----------------------------")
    print("Demand " , state.demand_list[i])
    state.update_state(i)
    print(state.print_state(i))

    

print(state.root.calculate_profit())
# print(state.rewards)
# print(state.root.plot_cost())
# print(state.plot_rewards())
