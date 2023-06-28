
import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
import matplotlib.pyplot as plt
import seaborn as sns 
import RLresult



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


# GA
# print('r1', state.run(54, 63, 42, 47, 42, 49))
ga = state.test_no_season(54, 63, 42, 47, 42, 49 )
ga_poisson = state.test_poisson_no_season(54, 63, 42, 47, 42, 49 )

# OLS
# print('r1', state.run(36, 44, 41, 48, 34, 38))
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
ols_poisson = state.test_poisson_no_season(36, 44, 41, 48, 34, 38)

ml = state.test_no_season(37, 41, 142, 149, 32, 35)
ml_poisson = state.test_poisson_no_season(37, 41, 142, 149, 32, 35)

#Reinforcement Learning
rl = RLresult.no_season()
rl_24 = RLresult.non_season_24()
rl_poisson =RLresult.poisson()

# print(bayesian)
# print(ga)
# print(ols)
# 



ax = sns.boxplot(data=[bayesian_poisson, ga_poisson, ols_poisson, rl_poisson, ml_poisson])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods, mean = %d', mean))
plt.show()

# ax = sns.boxplot(data=[bayesian, ga, ols, rl, ml])
# ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
# ax.set(xlabel='Methods', ylabel='Profit', title='Boxplot comparing Profits across Methods')
# plt.show()
