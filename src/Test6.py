from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand
from GenerateDemandMonthly import GenerateDemandMonthly

np.random.seed(1234)
demandable_state = [-1,0, 1, 1, 2, 2]
demand = GenerateDemandMonthly()

state = State()
state.create_state(demandable_state)

state.set_demand_list(demand.simulate_normal_no_season())
state.show_network()
print(state.demand_list)
""" for i in range(len(state.demand_list)):
    print("-----------------------------")
    print("Demand " , state.demand_list[i])
    state.update_state(i)
    print(state.print_state(i))

# print(state.root.calculate_profit())
# print(state.rewards)
# print(state.root.plot_cost())
print(state.root.plot_inv_level())
# print(state.plot_rewards())
 """