from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 



def validate(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    df = pd.read_csv("./src/TOTALSA.csv")
    mean = df['TOTALSA'].mean()
    std = df['TOTALSA'].std()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], amount=start_inventory, mean=mean, std=std)
    return (state.run(start_inventory, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1))
    

#x = validate(158, 71, 182, 67, 179, 80, 134)
x = validate(109.4, 59.58, 109, 49.24, 60.65, 43.36, 86.42)
print(x)