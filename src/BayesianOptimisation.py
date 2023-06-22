from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 


def BayesianOptimisation(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
        return -100000
    demand = GenerateDemandMonthly()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory)
    state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
    state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
    state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
    total_sum = 0
    df = pd.read_csv("./TOTALSA.csv")
    mean = df['TOTALSA'].mean()
    std = df['TOTALSA'].std()
    np.random.seed(1234)
    lst = np.reshape(demand.simulate_normal_no_season(periods = 24 * 100, mean=mean, std=std), (100, 24))
    for z in range(100):
        state.set_demand_list(lst[z])
        for i in range(24):
            # if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
            #         return -100000
            # state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
            # state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
            # state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
            state.update_state(i)
        total_sum += state.root.calculate_profit()
        state.reset(start_inventory)
    return total_sum / 100


