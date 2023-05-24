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

    def logic(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, demand):
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1],-100000000]
        amount = max(demand)
        state = MLState()
        state.create_state([-1 ,0, 1, 1, 2, 2], amount)
        state.change_demand(demand)
        state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
        state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
        state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
        for i in range(len(state.demand_list)):
            state.update_state(i)
        return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], state.total_sum()]

    def update_df(self, df, data):
        new_lst = data[0]
        new_lst.extend(data[1])
        new_lst.append(data[2])
        print(new_lst)
        df.loc[len(df.index)] = new_lst
        return df

    def create_data(self):
        print(Path.cwd())
        data = pd.read_csv('./src/data.csv')
        for i in data['Order_Demand']:
            elements = re.findall(r'\d+', i)
            demand = [int(x) for x in elements]
            if (len(demand) == 12):
                demand.sort()
                self.df = self.update_df(self.df, self.logic(random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), \
                    random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), demand))
                normal_demand = self.demand_generator.simulate_normal(1, demand[5])
                poisson_demand = self.demand_generator.simulate_poisson(1, demand[5])
                self.df = self.update_df(self.df, self.logic(random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), \
                        random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), normal_demand))
                self.df = self.update_df(self.df, self.logic(random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), \
                        random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), poisson_demand))



