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

#state.show_network()
print("length of demand list:", len(state.demand_list))
print(state.demand_list)

done = False
counter = 0
print(state.state)
while not done:
    """ print("-----------------------------")
    print("Demand " , state.demand_list[state.curr_time]) """
    intrand = np.random.randint(0, 8*8*8)
    RLstate, reward, done = state.step(intrand)
    print("Counter:", counter)
    print("Reward:", reward)
    print("done:", done)
    print("RLSTATE:", RLstate)
    print("-----------------------------")
    if counter == 30:
        break
    """ if counter > 0:
        print(state.print_state(counter))"""
    counter += 1 
    

# print(state.root.calculate_profit())
# print(state.rewards)
# print(state.root.plot_cost())
#print(state.root.plot_inv_level())
# print(state.plot_rewards())
 