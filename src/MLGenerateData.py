from Demandable import Demandable
from Item import Item
from MLState import MLState
import numpy as np
import pandas as pd
import re
import random
from GenerateDemandMonthly import GenerateDemandMonthly
from pathlib import Path

class MLGenerateData:
    def __init__(self):
        self.df = pd.DataFrame(columns=['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', \
        'month_11', 'month_12', 's_DC1', 'S_DC1', 's_DC2', 'S_DC2', 's_r1', 'S_r1', 'profit'])
        self.demand_generator = GenerateDemandMonthly()

    #convert to indiv col in df
    def logic_normal(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, std, runs):
        total = 0
        for x in range(runs):
            demand = self.demand_generator.simulate_normal(1, mean, std)
            amount = max(demand) * 1.5
            state = MLState()
            state.create_state([-1 ,0, 1, 1, 2, 2], amount)
            state.change_demand(demand)
            for i in range(len(state.demand_list)):
                if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
                    return None
                state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
                state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
                state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
                state.update_state(i)
            total += state.total_sum()
        return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total/runs]

    def logic_poisson(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, runs):
        total = 0
        for x in range(runs):
            demand = self.demand_generator.simulate_poisson(1, mean)
            amount = max(demand) * 1.5
            state = MLState()
            state.create_state([-1 ,0, 1, 1, 2, 2], amount)
            state.change_demand(demand)
            for i in range(len(state.demand_list)):
                if (s_DC1[i] >= S_DC1[i] or s_DC2[i] >= S_DC2[i] or s_r1[i] >= S_r1[i]):
                    return None
                state.changeable_network[0].change_order_point(round(s_r1[i]), round(S_r1[i]))
                state.changeable_network[1].change_order_point(round(s_DC1[i]), round(S_DC1[i]))
                state.changeable_network[2].change_order_point(round(s_DC2[i]), round(S_DC2[i]))
                state.update_state(i)
            total += state.total_sum()
        return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total/runs]

    def update_df(self, df, data):
        new_lst = data[0]
        new_lst.extend(data[1])
        new_lst.append(data[2])
        df.loc[len(df.index)] = new_lst
        return df

    def create_data(self):
        data = pd.read_csv('./src/data.csv')
        for i in data['Order_Demand']:
            elements = re.findall(r'\d+', i)
            demand = [int(x) for x in elements]
            if (len(demand) == 12):
                demand.sort()
                mean = np.mean(demand)
                std = np.std(demand)
                #print('demand: ' + str(demand))
                # log1 = self.logic(random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), \
                #     random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), demand)
                # if log1 is not None:
                #     self.df = self.update_df(self.df, log1)
                s = [round(x) for x in random.choices(range(round(np.quantile(demand, 0.1)), round(np.quantile(demand, 0.5))), k=12)]
                S = [round(x) for x in random.choices(range(round(np.quantile(demand, 0.5)), round(np.quantile(demand, 0.9))), k=12)]
                # print("s: " + str(s))
                # print("S: " + str(S))
                log2 = self.logic_normal(s, S, s, S, s, S, mean, std, 30)
                if log2 is not None:
                    self.df = self.update_df(self.df, log2)
                log3 = self.logic_poisson(s, S, s, S, s, S, mean, 30)
                if log3 is not None:
                    self.df = self.update_df(self.df, log3)




