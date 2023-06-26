from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 



def validate(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    df = pd.read_csv("./src/TOTALSA.csv")
    mean = df['TOTALSA'].mean()
    std = df['TOTALSA'].std()
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
    return (state.run(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1))
    

#x = validate(158, 71, 182, 67, 179, 80, 134)
x = validate(44.966674484487186, 52.68320959458202,  52.51721011581468, 77.96824363843548, 30.0, 57.224664841836486) 
print(x)