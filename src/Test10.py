from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

def validate(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    demand = GenerateDemandMonthly()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory)
    state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
    state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
    state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
    total_sum = 0
    np.random.seed(1234)
    for z in range(100):
        state.set_demand_list(demand.simulate_normal_no_season(mean=mean, std=std))
        print(state.demand_list)
        for i in range(24):
            # if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
            #         return -100000
            # state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
            # state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
            # state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
            state.update_state(i)
        total_sum += state.rewards
        state.reset(start_inventory)
    return total_sum / 100

# x = validate(158, 71, 182, 67, 179, 80, 134)
x = validate(58, 31,  85,  32, 121,  50,  78)
print(x)

# [31.0, 34.0, 32.0, 32.0, 32.0, 27.0, 27.0, 21.0, 30.0, 36.0, 27.0, 35.0, 29.0, 34.0, 36.0, 28.0, 22.0, 28.0, 35.0, 25.0, 30.0, 29.0, 31.0, 25.0]