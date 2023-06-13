from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand
from GenerateDemandMonthly import GenerateDemandMonthly

np.random.seed(1234)
demandable_state = [-1,0, 0]
demand = GenerateDemandMonthly()

state = State()
state.create_state(demandable_state)

#state.show_network()

print(state.demand_list)

done = False
while not done:
    print("-----------------------------")
    print("Demand " , state.demand_list[state.curr_time])
    RLstate, reward, done, info = state.step([30,20])
    print("Reward:", reward)
    print("done:", done)
    print("info:", info)
    print("RLSTATE:", RLstate)
    print("-----------------------------")
    

# print(state.root.calculate_profit())
# print(state.rewards)
# print(state.root.plot_cost())
#print(state.root.plot_inv_level())
# print(state.plot_rewards())
 