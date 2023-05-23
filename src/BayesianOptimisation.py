from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np

np.random.seed(1234)
def BayesianOptimisation(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    state = BayesianState()
    state.create_state([-1 ,0, 1, 1, 2, 2])
    state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
    state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
    state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
    for i in range(len(state.demand_list)):
        state.update_state(i)
    return state.total_sum()


