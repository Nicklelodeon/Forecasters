
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
bayesian_24 = state.test_no_season_24_period(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)


# GA
# print('r1', state.run(54, 63, 42, 47, 42, 49))
ga = state.test_no_season(54, 63, 42, 47, 42, 49)
ga_poisson = state.test_poisson_no_season(54, 63, 42, 47, 42, 49)
ga_24 = state.test_no_season_24_period(54, 63, 42, 47, 42, 49)

# OLS
# print('r1', state.run(36, 44, 41, 48, 34, 38))
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
ols_poisson = state.test_poisson_no_season(36, 44, 41, 48, 34, 38)
ols_24 = state.test_no_season_24_period(36, 44, 41, 48, 34, 38)

ml = state.test_no_season(37, 41, 142, 149, 32, 35)
ml_poisson = state.test_poisson_no_season(37, 41, 142, 149, 32, 35)
ml_24 = state.test_no_season_24_period(37, 41, 142, 149, 32, 35)

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
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods, mean = {}', round(mean, 2)))
plt.show()

print(np.mean(bayesian_poisson))
print(np.mean(ga_poisson))
print(np.mean(ols_poisson))
print(np.mean(rl_poisson))
print(np.mean(ml_poisson))

# print(np.mean(bayesian))
# print(np.mean(ga))
# print(np.mean(ols))
# print(np.mean(rl))
# print(np.mean(ml))

# ax = sns.boxplot(data=[bayesian, ga, ols, rl, ml])
# ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
# ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
# plt.show()


ax = sns.boxplot(data=[bayesian_24, ga_24, ols_24, rl_24, ml_24])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
plt.show()

print(np.mean(bayesian_24))
print(np.mean(ga_24))
print(np.mean(ols_24))
print(np.mean(rl_24))
print(np.mean(ml_24))