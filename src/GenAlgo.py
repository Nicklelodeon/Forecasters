from Demandable import Demandable
from Item import Item
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 

def Genetic_Algo_No_Season(array):
    start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1 = array
    if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
        return -100000
    iterations = 100
    demand = GenerateDemandMonthly()
    state = State()
    df = pd.read_csv("./TOTALSA.csv")
    mean = df['TOTALSA'].mean()
    std = df['TOTALSA'].std()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory)
    state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
    state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
    state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
    total_sum = 0
    np.random.seed(1234)
    for z in range(iterations):
        state.set_demand_list(demand.simulate_normal_no_season(mean = mean, std = std))
        for i in range(len(state.demand_list)):
            state.update_state(i)
        total_sum += state.rewards
        state.reset(start_inventory)
    return total_sum / iterations

def Genetic_Algo_Seasonal(array): #72 
    np.random.seed(1234)
    iterations = 100
    demand = GenerateDemandMonthly()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2])
    total_sum = 0
    for z in range(iterations):
        state.set_demand_list(demand.simulate_normal(1))
        for i in range(12):
            s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1 = array[6*i: 6*(i+1)]
            state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
            state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
            state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
            state.update_state(i)
        total_sum += state.rewards
        state.reset()
    return total_sum / iterations
        