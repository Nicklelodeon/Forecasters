from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 



def validate(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
<<<<<<< HEAD
    iterations = 1000
    periods = 24
    demand = GenerateDemandMonthly()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory)
    state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
    state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
    state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
    lst = np.zeros(iterations)
    np.random.seed(1234)
    lst = np.reshape(demand.simulate_normal_no_season(periods = periods * iterations, mean=mean, std=std), (iterations, periods))
    for z in range(iterations):
        state.set_demand_list(lst[z])
        #print(state.demand_list)
        for i in range(periods):
            # if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
            #         return -100000
            # state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
            # state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
            # state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
            state.update_state(i)
        lst[z] = state.rewards
        #total_sum += state.rewards
        state.reset(start_inventory)
    return ("Mean result: {} \n Std result: {}".format(np.mean(lst), np.std(lst)))
    #return total_sum / 100
=======
    df = pd.read_csv("./src/TOTALSA.csv")
    mean = df['TOTALSA'].mean()
    std = df['TOTALSA'].std()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory, mean=mean, std=std, cost=4)
    return (state.run(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1))
    
>>>>>>> refs/remotes/origin/main

#x = validate(158, 71, 182, 67, 179, 80, 134)
x = validate(86, 55, 64, 86, 126, 53, 58)
print(x)