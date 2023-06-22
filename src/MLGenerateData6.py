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
        self.state.create_state([-1 ,0, 1, 1, 2, 2])


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
    def logic_normal(self, start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, std):
        total = 0
        s_r1 = round(s_r1)
        S_r1 = round(S_r1)
        s_DC1 = round(s_DC1) 
        S_DC1 = round(S_DC1)
        s_DC2 = round(s_DC2)
        S_DC2 = round(S_DC2)
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        
        for x in range(100):
            demand = self.demand_generator.simulate_normal_no_season(mean= mean, std = std, periods=108)
            
            self.state.reset(start_inventory)
            self.state.set_demand_list(demand)
            self.state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
            self.state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
            self.state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
            for i in range(len(self.state.demand_list)):
                self.state.update_state(i)
            total += self.state.calculate_profits()
        self.df = self.update_df(self.df, [[mean, std, 0], [start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total/100])
        

    def logic_poisson(self, start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean):
        total = 0
        s_r1 = round(s_r1)
        S_r1 = round(S_r1)
        s_DC1 = round(s_DC1) 
        S_DC1 = round(S_DC1)
        s_DC2 = round(s_DC2)
        S_DC2 = round(S_DC2)
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return None
        for x in range(100):
            demand = self.demand_generator.simulate_poisson_no_season(mean= mean, periods=108)
            self.state.reset(start_inventory)
            self.state.set_demand_list(demand)
            self.state.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
            self.state.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
            self.state.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
            for i in range(len(self.state.demand_list)):
                self.state.update_state(i)
            total += self.state.calculate_profits()
        self.df = self.update_df(self.df, [[mean, 0, 1], [start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1], total/100])

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
        lst = ['mean', 'std', 'distribution']
        lst.extend(["start_inventory", "s_DC1", "S_DC1", "s_DC2", "S_DC2", "s_r1", "S_r1", "profit"])
        self.df = pd.concat([self.df, pd.DataFrame(columns=lst)])
        
    
    def create_data(self):
        self.create_df_col_names()
        data = pd.read_csv('src/US_cleaned_car_data.csv')
        all_demand = []
        count = 1
        for i in data.columns[1:]:
            demand = [x for x in data[i]]
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

            for x in range(50):
                self.logic_normal(random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), mean, std)
                # if log2 is not None:
                #     self.df = self.update_df(self.df, log2)
                self.logic_poisson(random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), mean)
                    # if log3 is not None:
                    #     self.df = self.update_df(self.df, log3)
                


data = MLGenerateData6()
data.create_data()
# data.df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/mldata.csv")

data.df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/6_24months_US_car_data_try_50.csv")
