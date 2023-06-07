from Demandable import Demandable
from Item import Item
from MLState import MLState
import numpy as np
import pandas as pd
import re
import random
from GenerateDemandMonthly import GenerateDemandMonthly
from pathlib import Path

class MLGenerateData6:
    def __init__(self):
        
        self.df = pd.DataFrame()
        self.demand_generator = GenerateDemandMonthly()
        self.state = MLState()
        self.state.create_state([-1 ,0, 1, 1, 2, 2])


    def logic(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, demand):
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        amount = max(demand) * 3
        self.state.reset(amount)
        self.state.change_demand(demand)
        self.state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
        self.state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
        self.state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
        for i in range(len(self.state.demand_list)):
            if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
                return None
            self.state.update_state(i)
        return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], self.state.total_sum()]
        

    #convert to indiv col in df
    def logic_normal(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, std, runs):
        total = 0
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        for x in range(runs):
            demand = self.demand_generator.simulate_normal_no_season(mean= mean, std = std)
            amount = max(demand) * 3
            self.state.reset(amount)
            self.state.change_demand(demand)
            self.state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
            self.state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
            self.state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
            for i in range(len(self.state.demand_list)):
                self.state.update_state(i)
            total += self.state.total_sum()
        return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total/runs]

    def logic_poisson(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, runs):
        total = 0
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        for x in range(runs):
            demand = self.demand_generator.simulate_poisson_no_season(mean = mean)
            amount = max(demand) * 1.5
            self.state.reset(amount)
            self.state.change_demand(demand)
            self.state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
            self.state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
            self.state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
            for i in range(len(self.state.demand_list)):
                self.state.update_state(i)
            total += self.state.total_sum()
        return [demand, [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total/runs]

    def update_df(self, df, data):
        new_lst = []
        new_lst.extend(data[0])
        new_lst.extend(data[1])
        new_lst.append(data[2])
        print(new_lst)
        df.loc[len(df.index)] = new_lst
        return df

    def create_df_cols(self, string):
        return [string + str(x) for x in range(1, 25)]

    def create_df_col_names(self):
        lst = self.create_df_cols("month")
        lst.extend(["s_DC1", "S_DC1", "s_DC2", "S_DC2", "s_r1", "S_r1", "profit"])
        self.df = pd.concat([self.df, pd.DataFrame(columns=lst)])
        
    
    def create_data(self):
        self.create_df_col_names()
        data = pd.read_csv('src/cleaned_car_data.csv')
        all_demand = []
        count = 1
        print(data.columns[1:])
        for i in data.columns[1:]:
            demand = [x for x in data[i]]
            all_demand.extend(demand)
            mean = np.mean(demand)
            std = np.std(demand)
            if count % 2 == 0:
                log1 = self.logic(random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), all_demand)
                if log1 is not None:
                    self.df = self.update_df(self.df, log1)
                all_demand = []
            #print('demand: ' + str(demand))
            # log1 = self.logic(random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), \
            #     random.randint(demand[9], demand[11]), random.randint(demand[1], demand[3]), random.randint(demand[9], demand[11]), demand)
            # if log1 is not None:
            #     self.df = self.update_df(self.df, log1)

            # print("s: " + str(s))
            # print("S: " + str(S))
            # log1 = self.logic(random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), demand)
            # if log1 is not None:
            #     self.df = self.update_df(self.df, log1)
            for x in range(100):
                log2 = self.logic_normal(random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), mean, std, 100)
                if log2 is not None:
                    self.df = self.update_df(self.df, log2)
                log3 = self.logic_poisson(random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), mean, 100)
                if log3 is not None:
                    self.df = self.update_df(self.df, log3)
            count += 1


data = MLGenerateData6()
data.create_data()
# data.df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/mldata.csv")

data.df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/6_24months_car_data.csv")
