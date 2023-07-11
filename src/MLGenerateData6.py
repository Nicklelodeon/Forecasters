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

        

    
    def logic_normal(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1, mean, std):
        """updates dataframe with random samples of (s, S) policies, mean, std and their corresponding profits

        Args:
            s_DC1 (int): s_DC1 
            S_DC1 (int): 
            s_DC2 (int): _description_
            S_DC2 (int): _description_
            s_r1 (int): _description_
            S_r1 (int): _description_
            mean (float): mean of demand distribution
            std (float): std of demand distribution
        """
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
        

    def update_df(self, df, data):
        """updates df with data provided

        Args:
            df (DataFrame): existing dataframe
            data (list): data to be added

        Returns:
            DataFrame: updated df
        """
        new_lst = []
        new_lst.extend(data[0])
        new_lst.extend(data[1])
        new_lst.append(data[2])
        df.loc[len(df.index)] = new_lst
        return df

    def create_df_cols(self, string):
        return [string + str(x) for x in range(1, 109)]

    def create_df_col_names(self):
        """creates col names of df
        """
        lst = ['mean', 'std']
        lst.extend(["s_DC1", "S_DC1", "s_DC2", "S_DC2", "s_r1", "S_r1", "profit"])
        self.df = pd.concat([self.df, pd.DataFrame(columns=lst)])
        
    
    def create_data(self):
        """generate mean and std based on data in csv file, then simulate each mean and std with 10000 different (s, S) policy
        """
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
            for x in range(10000):
                self.logic_normal(random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), mean, std)
                
