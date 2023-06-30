from Demandable import Demandable
from Item import Item
from State import State
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
        self.state = State()
        df = pd.read_csv("./src/TOTALSA.csv")
        mean = df['TOTALSA'].mean()
        std = df['TOTALSA'].std()
        self.state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)


    def logic(self, start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, demand):
        s_r1 = round(s_r1)
        S_r1 = round(S_r1)
        s_DC1 = round(s_DC1) 
        S_DC1 = round(S_DC1)
        s_DC2 = round(s_DC2)
        S_DC2 = round(S_DC2)
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        self.state.set_demand_matrix(demand)
        print(demand)
        total = self.state.run(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1)
        return [demand, [start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total]
        

    #convert to indiv col in df
    def logic_normal(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, std):
        s_r1 = round(s_r1)
        S_r1 = round(S_r1)
        s_DC1 = round(s_DC1) 
        S_DC1 = round(S_DC1)
        s_DC2 = round(s_DC2)
        S_DC2 = round(S_DC2)
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return
        self.state.set_demand_matrix(self.state.create_normal_demand(mean=mean, std=std))
        profit = self.state.run(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1)
        self.df = self.update_df(self.df, [[mean, std], [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], profit])
        

    def logic_poisson(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean):
        s_r1 = round(s_r1)
        S_r1 = round(S_r1)
        s_DC1 = round(s_DC1) 
        S_DC1 = round(S_DC1)
        s_DC2 = round(s_DC2)
        S_DC2 = round(S_DC2)
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        self.state.set_demand_matrix(self.state.create_poisson_demand(mean=mean))
        profit = self.state.run(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1)
        self.df = self.update_df(self.df, [[mean, mean**0.5, 1], [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], profit])

    def update_df(self, df, data):
        new_lst = []
        new_lst.extend(data[0])
        new_lst.extend(data[1])
        new_lst.append(data[2])
        df.loc[len(df.index)] = new_lst
        return df

    def create_df_cols(self, string):
        return [string + str(x) for x in range(1, 109)]

    def create_df_col_names(self):
        lst = ['mean', 'std']
        lst.extend(["s_DC1", "S_DC1", "s_DC2", "S_DC2", "s_r1", "S_r1", "profit"])
        self.df = pd.concat([self.df, pd.DataFrame(columns=lst)])
        
    
    def create_data(self):
        self.create_df_col_names()
        data = pd.read_csv('src/US_cleaned_car_data.csv')
        all_demand = []
        count = 1
        for i in data.columns[1:]:
            demand = [x for x in data[i]]
            print(demand)
            all_demand.extend(demand)
            mean = np.mean(all_demand)
            std = np.std(all_demand)
                # log1 = self.logic(random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), all_demand)
                # if log1 is not None:
                #     self.df = self.update_df(self.df, log1)
                
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

            for x in range(500):
                self.logic_normal(random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), mean, std)
                # if log2 is not None:
                #     self.df = self.update_df(self.df, log2)
                # self.logic_poisson(random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), mean)
                    # if log3 is not None:
                    #     self.df = self.update_df(self.df, log3)
                


data = MLGenerateData6()
data.create_data()
# data.df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/mldata.csv")

data.df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/6_24months_US_car_data_try_50.csv")
