from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
import GenerateDemandMonthly 

np.random.seed(1234)

def BayesianOptimisation(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    demand = GenerateDemandMonthly()
    state = BayesianState()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory)
    total_sum = 0
    for z in range(30):
        state.set_demand_list(demand.simulate_normal())
        for i in range(12):
            if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
                    return -100000
            state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
            state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
            state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
            state.update_state(i)
        total_sum += state.rewards
        state.reset(start_inventory)
    return total_sum / 30


