import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
import matplotlib.pyplot as plt
import seaborn as sns 
import RLtest

from scipy import stats

#### Generate New Demand ####


df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()
state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# Bayesian
# normal demand distribution with 108 time periods, floored triangle lead time distribution
bayesian = state.test_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
# Poisson demand distribution with 108 time periods, floored triangle lead time distribution
bayesian_poisson = state.test_poisson_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
# normal demand distribution with 24 time periods, floored triangle lead time distribution
bayesian_24 = state.test_no_season_24_period(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
# real data
bayesian_real = state.test_real_data(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
# Normal demand distribution with 108 time periods, shifted Poisson lead time distribution
bayesian_poisson_lead_time = state.test_no_season_poisson_lead_time(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
# Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution
bayesian_both_poisson = state.test_poisson_no_season_poisson_lead_time(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)

state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# GA
# normal demand distribution with 108 time periods, floored triangle lead time distribution
ga = state.test_no_season(54, 63, 42, 47, 42, 49)
# Poisson demand distribution with 108 time periods, floored triangle lead time distribution
ga_poisson = state.test_poisson_no_season(54, 63, 42, 47, 42, 49)
# normal demand distribution with 24 time periods, floored triangle lead time distribution
ga_24 = state.test_no_season_24_period(54, 63, 42, 47, 42, 49)
# real data
ga_real = state.test_real_data(54, 63, 42, 47, 42, 49)
# normal demand distribution with 108 time periods, shifted Poisson lead time distribution
ga_poisson_lead_time = state.test_no_season_poisson_lead_time(54, 63, 42, 47, 42, 49)
# Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution
ga_both_poisson = state.test_poisson_no_season_poisson_lead_time(54, 63, 42, 47, 42, 49)



state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# OLS
# normal demand distribution with 108 time periods, floored triangle lead time distribution 
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
# Poisson demand distribution with 108 time periods, floored triangle lead time distribution
ols_poisson = state.test_poisson_no_season(36, 44, 41, 48, 34, 38)
# normal demand distribution with 24 time periods, floored triangle lead time distribution
ols_24 = state.test_no_season_24_period(36, 44, 41, 48, 34, 38)
# real data
ols_real = state.test_real_data(36, 44, 41, 48, 34, 38)
# normal demand distribution with 108 time periods, shifted Poisson lead time distribution
ols_poisson_lead_time =  state.test_no_season_poisson_lead_time(36, 44, 41, 48, 34, 38)
# Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution
ols_both_poisson = state.test_poisson_no_season_poisson_lead_time(36, 44, 41, 48, 34, 38)


state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
# ML
# normal demand distribution with 108 time periods, floored triangle lead time distribution
ml = state.test_no_season(37, 41, 142, 149, 32, 35)
# Poisson demand distribution with 108 time periods, floored triangle lead time distribution
ml_poisson = state.test_poisson_no_season(37, 41, 142, 149, 32, 35)
# normal demand distribution with 24 time periods, floored triangle lead time distribution
ml_24 = state.test_no_season_24_period(37, 41, 142, 149, 32, 35)
# real data
ml_real = state.test_real_data(37, 41, 142, 149, 32, 35)
# normal demand distribution with 108 time periods, shifted Poisson lead time distribution
ml_poisson_lead_time = state.test_no_season_poisson_lead_time(37, 41, 142, 149, 32, 35)
# Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution
ml_both_poisson = state.test_poisson_no_season_poisson_lead_time(37, 41, 142, 149, 32, 35)


state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# Reinforcement Learning
# normal demand distribution with 108 time periods, floored triangle lead time distribution
rl = RLtest.test_no_season()
# normal demand distribution with 24 time periods, floored triangle lead time distribution
rl_24 = RLtest.test_no_season_24_period()
# Poisson demand distribution with 108 time periods, floored triangle lead time distribution
rl_poisson =RLtest.test_poisson_no_season()
# real data
rl_real = RLtest.test_real_data()
# normal demand distribution with 108 time periods, shifted Poisson lead time distribution
rl_poisson_lead_time = RLtest.test_no_season_poisson_lead_time()
# Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution
rl_both_poisson = RLtest.test_poisson_no_season_poisson_lead_time()



state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# Random search
# normal demand distribution with 108 time periods, floored triangle lead time distribution
random = state.test_no_season(55, 70, 70, 75, 41, 45)
# Poisson demand distribution with 108 time periods, floored triangle lead time distribution
random_poisson = state.test_poisson_no_season(55, 70, 70, 75, 41, 45)
# normal demand distribution with 24 time periods, floored triangle lead time distribution
random_24 = state.test_no_season_24_period(55, 70, 70, 75, 41, 45)
# real data
random_real = state.test_real_data(55, 70, 70, 75, 41, 45)
# normal demand distribution with 108 time periods, shifted Poisson lead time distribution
random_poisson_lead_time =  state.test_no_season_poisson_lead_time(55, 70, 70, 75, 41, 45)
# Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution
random_both_poisson = state.test_poisson_no_season_poisson_lead_time(55, 70, 70, 75, 41, 45)

labels = ['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Best Random']
colors = {'Bayesian': 'blue', 'GA': 'orange', 'OLS': 'green', 'RL': 'red', 'ML': 'purple', 'Best Random': 'brown'}


# plot real data
y_1 = [bayesian_real, ga_real, ols_real, rl_real, ml_real, random_real]
x_1 = labels
df1 = pd.DataFrame({"y":y_1,
                   "x":x_1})

ax = sns.barplot(data=df1, x="x", y="y")
ax.set(xlabel='Methods', ylabel='Profit', title='Using real monthly data from figure 4.1')
plt.show()

# plot boxplot for normal demand distribution with 108 time periods, floored triangle lead time distribution 
ax = sns.boxplot(data=[bayesian, ga, ols, rl, ml, random], showfliers=False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods,\nmean = {}, std = {} with floored triangle lead time distribution', round(mean, 2), round(std, 2)))
plt.show()

# plot error bar for normal demand distribution with 108 time periods, floored triangle lead time distribution 
tick_labels = labels
ax = sns.pointplot(data = [bayesian, ga, ols, rl, ml, random],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods,\nmean = {}, std = {} with floored triangle lead time distribution', round(mean, 2), round(std, 2)))
plt.show()

# plot boxplot for Poisson demand distribution with 108 time periods, floored triangle lead time distribution 
ax = sns.boxplot(data=[bayesian_poisson, ga_poisson, ols_poisson, rl_poisson, ml_poisson, random_poisson], showfliers=False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods,\nmean = {} with floored triangle lead time distribution', round(mean, 2)))
plt.show()

# plot error bar for Poisson demand distribution with 108 time periods, floored triangle lead time distribution 
tick_labels = labels
ax = sns.pointplot(data = [bayesian_poisson, ga_poisson, ols_poisson, rl_poisson, ml_poisson, random_poisson],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods, standard error of 2 std,\n mean = {} with floored triangle lead time distribution', round(mean, 2)))
plt.show()

# plot boxplot for normal demand distribution with 24 time periods, floored triangle lead time distribution 
ax = sns.boxplot(data=[bayesian_24, ga_24, ols_24, rl_24, ml_24, random_24], showfliers=False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods,\nmean = {}, std = {} with floored triangle lead time distribution', round(mean, 2), round(std, 2)))
plt.show()

# plot error bar for normal demand distribution with 24 time periods, floored triangle lead time distribution 
tick_labels = labels
ax = sns.pointplot(data=[bayesian_24, ga_24, ols_24, rl_24, ml_24, random_24],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])

ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods,standard error of 2 std, \nmean = {}, std = {} with floored triangle lead time distribution', round(mean, 2), round(std, 2)))
plt.show()

# plot boxplot for normal demand distribution with 108 time periods, shifted Poisson lead time distribution 
tick_labels = labels
ax = sns.boxplot(data=[bayesian_poisson_lead_time, ga_poisson_lead_time, ols_poisson_lead_time,
                       rl_poisson_lead_time, ml_poisson_lead_time, random_poisson_lead_time], showfliers = False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods,\nmean = {}, std = {}, with shifted Poisson lead time', round(mean, 2), round(std, 2)))
plt.show()

# plot error bar for normal demand distribution with 108 time periods, shifted Poisson lead time distribution 
ax = sns.pointplot(data=[bayesian_poisson_lead_time, ga_poisson_lead_time, ols_poisson_lead_time,
                       rl_poisson_lead_time, ml_poisson_lead_time, random_poisson_lead_time],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])

ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods, standard error of 2 std, \nmean = {}, std = {}, with shifted Poisson lead time', round(mean, 2), round(std, 2)))
plt.show()

# plot boxplot for Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution 
ax = sns.boxplot(data=[bayesian_both_poisson, ga_both_poisson, ols_both_poisson,
                       rl_both_poisson, ml_both_poisson, random_both_poisson], showfliers = False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods,\nmean = {}, with shifted Poisson lead time', round(mean, 2)))
plt.show()

# plot error bar for Poisson demand distribution with 108 time periods, shifted Poisson lead time distribution 
ax = sns.pointplot(data=[bayesian_both_poisson, ga_both_poisson, ols_both_poisson,
                       rl_both_poisson, ml_both_poisson, random_both_poisson],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])

ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods, standard error of 2 std, \n mean = {}, with shifted Poisson lead time', round(mean, 2)))
plt.show()