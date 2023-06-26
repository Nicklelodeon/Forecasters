import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly


#### Generate New Demand ####


df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()
state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)


print('r1', state.run(34.74, 148.2, 64.22 , 103.2, 44.54,  119.5 ))
print('r2', state.test_no_season(34.74, 148.2, 64.22 , 103.2, 44.54,  119.5 ))


state.test_no_season(51, 52, 44, 58, 45, 47)