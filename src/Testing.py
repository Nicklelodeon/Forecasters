import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
import matplotlib.pyplot as plt
import seaborn as sns 
# import sys
# sys.path.append("RLsrc/")
# import RLtest



#### Generate New Demand ####


df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()
state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# Bayesian
# print('r1', state.run(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117))
bayesian = state.test_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
bayesian_poisson = state.test_poisson_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(np.mean(bayesian_poisson))

# GA
# print('r1', state.run(54, 63, 42, 47, 42, 49))
ga = state.test_no_season(54, 63, 42, 47, 42, 49 )

# OLS
# print('r1', state.run(36, 44, 41, 48, 34, 38))
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
# print(bayesian)
# print(ga)
# print(ols)
# 




