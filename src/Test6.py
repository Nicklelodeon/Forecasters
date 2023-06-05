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

demandable_state = [-1,0, 1, 1, 2, 2]

state = State()
state.create_state(demandable_state)
<<<<<<< HEAD
=======
#state.show_network()
>>>>>>> refs/remotes/origin/main
state.set_demand_list(demand_list)
print(state.demand_list)
for i in range(len(state.demand_list)):
    print("-----------------------------")
    print("Demand " , state.demand_list[i])
    state.update_state(i)
    print(state.print_state(i))

<<<<<<< HEAD
print(state.root.calculate_profit())
print(state.rewards)
#78094
=======
# print(state.root.calculate_profit())
# print(state.rewards)
# print(state.root.plot_cost())
print(state.root.plot_inv_level())
# print(state.plot_rewards())
>>>>>>> refs/remotes/origin/main
