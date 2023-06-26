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

# Bayesian
# print('r1', state.run(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117))
# print('r2', state.test_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117))

print('r1', state.run(64, 91, 60, 93, 35, 41))
print('r1', state.test_no_season(64, 91, 60, 93, 35, 41))



# state.run_concurrent(51, 52, 44, 58, 45, 47)
