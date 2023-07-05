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
bayesian_real = state.test_real_data(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)

# GA
# print('r1', state.run(54, 63, 42, 47, 42, 49))
ga = state.test_no_season(54, 63, 42, 47, 42, 49)
ga_poisson = state.test_poisson_no_season(54, 63, 42, 47, 42, 49)
ga_24 = state.test_no_season_24_period(54, 63, 42, 47, 42, 49)
ga_real = state.test_real_data(54, 63, 42, 47, 42, 49)

# OLS
# print('r1', state.run(36, 44, 41, 48, 34, 38))
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
ols_poisson = state.test_poisson_no_season(36, 44, 41, 48, 34, 38)
ols_24 = state.test_no_season_24_period(36, 44, 41, 48, 34, 38)
ols_real = state.test_real_data(36, 44, 41, 48, 34, 38)

ml = state.test_no_season(37, 41, 142, 149, 32, 35)
ml_poisson = state.test_poisson_no_season(37, 41, 142, 149, 32, 35)
ml_24 = state.test_no_season_24_period(37, 41, 142, 149, 32, 35)
ml_real = state.test_real_data(37, 41, 142, 149, 32, 35)

#Reinforcement Learning
rl = RLresult.no_season()
rl_24 = RLresult.non_season_24()
rl_poisson =RLresult.poisson()
rl_real = 554512.7999999998


# Random search
random = state.test_no_season(55, 70, 70, 75, 41, 45)
random_poisson = state.test_poisson_no_season(55, 70, 70, 75, 41, 45)
random_24 = state.test_no_season_24_period(55, 70, 70, 75, 41, 45)
random_real = state.test_real_data(55, 70, 70, 75, 41, 45)

## Random Result:
print("random normal", np.mean(random))
print("random poisson", np.mean(random_poisson))
print("random normal 24", np.mean(random_24))

print("real bayesian:", bayesian_real)
print("ga real", ga_real)
print("ols real", ols_real)
print("rl real", 554512.7999999998)
print("ml real:", ml_real)
print("random real", random_real)

ax = sns.boxplot(data=[bayesian, ga, ols, rl, ml, random])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
plt.show()

tick_labels = ['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random']
colors = {'Bayesian': 'blue', 'GA': 'orange', 'OLS': 'green', 'RL': 'red', 'ML': 'purple', 'Random': 'brown'}
ax = sns.pointplot(data = [bayesian, ga, ols, rl, ml, random],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
plt.show()


y_1 = [bayesian_real, ga_real, ols_real, rl_real, ml_real, random_real]
x_1 = ['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random']
df = pd.DataFrame({"y":y_1,
                   "x":x_1})

ax = sns.barplot(data=df, x="x", y="y")
ax.set(xlabel='Methods', ylabel='Profit', title='Using real monthly data from figure 4.1')


#ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
plt.show()

# df_108 = pd.DataFrame( {'bayesian': bayesian,
#                         'GA': ga,
#                         'OLS': ols,
#                         'ML': ml,
#                         'RL': rl,
#                         'Random': random})
# print(df_108)

ax = sns.boxplot(data=[bayesian_poisson, ga_poisson, ols_poisson, rl_poisson, ml_poisson, random_poisson])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods, mean = {}', round(mean, 2)))
plt.show()

tick_labels = ['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random']
colors = {'Bayesian': 'blue', 'GA': 'orange', 'OLS': 'green', 'RL': 'red', 'ML': 'purple', 'Random': 'brown'}
ax = sns.pointplot(data = [bayesian_poisson, ga_poisson, ols_poisson, rl_poisson, ml_poisson, random_poisson],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods,\nstandard error of 2 std, mean = {}', round(mean, 2)))
plt.show()
# blue orange green red purple
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


ax = sns.boxplot(data=[bayesian_24, ga_24, ols_24, rl_24, ml_24, random_24])
ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods, mean = {}, std = {}', round(mean, 2), round(std, 2)))
plt.show()

tick_labels = ['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random']
colors = {'Bayesian': 'blue', 'GA': 'orange', 'OLS': 'green', 'RL': 'red', 'ML': 'purple', 'Random': 'brown'}
ax = sns.pointplot(data=[bayesian_24, ga_24, ols_24, rl_24, ml_24, random_24],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])

ax.set_xticklabels(['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Random'])
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods,\nstandard error of 2 std, mean = {}, std = {}', round(mean, 2), round(std, 2)))
plt.show()

print(np.mean(bayesian_24))
print(np.mean(ga_24))
print(np.mean(ols_24))
print(np.mean(rl_24))
print(np.mean(ml_24))