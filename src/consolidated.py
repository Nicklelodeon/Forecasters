import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly
import matplotlib.pyplot as plt
import seaborn as sns 
import RLresult

from scipy import stats

#### Generate New Demand ####


df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()
state = State()
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# Bayesian
print("Bayesian")
bayesian_og = state.run(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(stats.sem(bayesian_og))
bayesian = state.test_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(stats.sem(bayesian))
bayesian_poisson = state.test_poisson_no_season(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(stats.sem(bayesian_poisson))
bayesian_24 = state.test_no_season_24_period(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(stats.sem(bayesian_24))
bayesian_real = state.test_real_data(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(stats.sem(bayesian_real))
bayesian_poisson_lead_time = state.test_no_season_poisson_lead_time(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print(stats.sem(bayesian_poisson_lead_time))
bayesian_both_poisson = state.test_poisson_no_season_poisson_lead_time(42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117)
print("BAYE",stats.sem(bayesian_both_poisson))
state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# GA
print("Genetic Algorithm")
ga_og = state.run(54, 63, 42, 47, 42, 49)
print(stats.sem(ga_og))
ga = state.test_no_season(54, 63, 42, 47, 42, 49)
print(stats.sem(ga))
ga_poisson = state.test_poisson_no_season(54, 63, 42, 47, 42, 49)
print(stats.sem(ga_poisson))
ga_24 = state.test_no_season_24_period(54, 63, 42, 47, 42, 49)
print(stats.sem(ga_24))
ga_real = state.test_real_data(54, 63, 42, 47, 42, 49)
print(stats.sem(ga_real))
ga_poisson_lead_time = state.test_no_season_poisson_lead_time(54, 63, 42, 47, 42, 49)
print(stats.sem(ga_poisson_lead_time))
ga_both_poisson = state.test_poisson_no_season_poisson_lead_time(54, 63, 42, 47, 42, 49)
print("GA",stats.sem(ga_both_poisson))


state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

# OLS
print("OLS")
ols_og = state.run(36, 44, 41, 48, 34, 38)
print(stats.sem(ols_og))
ols = state.test_no_season(36, 44, 41, 48, 34, 38)
print(stats.sem(ols))
ols_poisson = state.test_poisson_no_season(36, 44, 41, 48, 34, 38)
print(stats.sem(ols_poisson))
ols_24 = state.test_no_season_24_period(36, 44, 41, 48, 34, 38)
print(stats.sem(ols_24))
ols_real = state.test_real_data(36, 44, 41, 48, 34, 38)
print(stats.sem(ols_real))
ols_poisson_lead_time =  state.test_no_season_poisson_lead_time(36, 44, 41, 48, 34, 38)
print(stats.sem(ols_poisson_lead_time))
ols_both_poisson = state.test_poisson_no_season_poisson_lead_time(36, 44, 41, 48, 34, 38)
print("OLS",stats.sem(ols_both_poisson))

state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)

ml_og = state.run(37, 41, 142, 149, 32, 35)
print(stats.sem(ml_og))
ml = state.test_no_season(37, 41, 142, 149, 32, 35)
print(stats.sem(ml))
ml_poisson = state.test_poisson_no_season(37, 41, 142, 149, 32, 35)
print(stats.sem(ml_poisson))
ml_24 = state.test_no_season_24_period(37, 41, 142, 149, 32, 35)
print(stats.sem(ml_24))
ml_real = state.test_real_data(37, 41, 142, 149, 32, 35)
print(stats.sem(ml_real))
ml_poisson_lead_time = state.test_no_season_poisson_lead_time(37, 41, 142, 149, 32, 35)
print(stats.sem(ml_poisson_lead_time))
ml_both_poisson = state.test_poisson_no_season_poisson_lead_time(37, 41, 142, 149, 32, 35)
print("ML",stats.sem(ols_both_poisson))

state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
# Reinforcement Learning
rl = RLresult.no_season()
rl_24 = RLresult.non_season_24()
rl_poisson =RLresult.poisson()
rl_real = 554512.7999999998
rl_poisson_lead_time = RLresult.poisson()
rl_both_poisson = RLresult.both_poisson()
print("RL sem poisson",stats.sem(rl_poisson_lead_time))
print("RL sem both poisson", stats.sem(rl_both_poisson))


state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
# Random search
random_og = state.run(55, 70, 70, 75, 41, 45)
print(stats.sem(random_og))
random = state.test_no_season(55, 70, 70, 75, 41, 45)
print(stats.sem(random))
random_poisson = state.test_poisson_no_season(55, 70, 70, 75, 41, 45)
print(stats.sem(random_poisson))
random_24 = state.test_no_season_24_period(55, 70, 70, 75, 41, 45)
print(stats.sem(random_24))
random_real = state.test_real_data(55, 70, 70, 75, 41, 45)
print(stats.sem(random_real))
random_poisson_lead_time =  state.test_no_season_poisson_lead_time(55, 70, 70, 75, 41, 45)
print(stats.sem(random_poisson_lead_time))
random_both_poisson = state.test_poisson_no_season_poisson_lead_time(55, 70, 70, 75, 41, 45)
print("RANDOM",stats.sem(random_both_poisson))

# print("real bayesian:", bayesian_real)
# print("ga real", ga_real)
# print("ols real", ols_real)
# print("ml real:", ml_real)
# print("random real", random_real)

# # df_108 = pd.DataFrame( {'bayesian': bayesian,
# #                         'GA': ga,
# #                         'OLS': ols,
# #                         'ML': ml,
# #                         'RL': rl,
# #                         'Best Random': random})
# # print(df_108)

labels = ['Bayesian', 'GA', 'OLS', 'RL', 'ML', 'Best Random']
colors = {'Bayesian': 'blue', 'GA': 'orange', 'OLS': 'green', 'RL': 'red', 'ML': 'purple', 'Best Random': 'brown'}


# y_1 = [bayesian_real, ga_real, ols_real, rl_real, ml_real, random_real]
# x_1 = labels
# df1 = pd.DataFrame({"y":y_1,
#                    "x":x_1})

# ax = sns.barplot(data=df1, x="x", y="y")
# ax.set(xlabel='Methods', ylabel='Profit', title='Using real monthly data from figure 4.1')
# plt.show()

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

ax = sns.boxplot(data=[bayesian_poisson, ga_poisson, ols_poisson, rl_poisson, ml_poisson, random_poisson], showfliers=False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods,\nmean = {} with floored triangle lead time distribution', round(mean, 2)))
plt.show()

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

ax = sns.boxplot(data=[bayesian, ga, ols, rl, ml, random], showfliers=False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods,\nmean = {}, std = {} with floored triangle lead time distribution', round(mean, 2), round(std, 2)))
plt.show()


ax = sns.boxplot(data=[bayesian_24, ga_24, ols_24, rl_24, ml_24, random_24], showfliers=False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 24 periods,\nmean = {}, std = {} with floored triangle lead time distribution', round(mean, 2), round(std, 2)))
plt.show()

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

y_1 = [bayesian_real, ga_real, ols_real, rl_real, ml_real, random_real]
x_1 = labels
df = pd.DataFrame({"y":y_1,
                   "x":x_1})

ax = sns.barplot(data=df, x="x", y="y")
ax.set(xlabel='Methods', ylabel='Profit', title='Using real monthly data from figure 4.1')
plt.show()

tick_labels = labels

print('mean poisson lead time')
print(np.mean(bayesian_poisson_lead_time))
print(np.mean(ga_poisson_lead_time))
print(np.mean(ols_poisson_lead_time))
print(np.mean(rl_poisson_lead_time))
print(np.mean(ml_poisson_lead_time))
print(np.mean(random_poisson_lead_time))

ax = sns.boxplot(data=[bayesian_poisson_lead_time, ga_poisson_lead_time, ols_poisson_lead_time,
                       rl_poisson_lead_time, ml_poisson_lead_time, random_poisson_lead_time], showfliers = False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods,\nmean = {}, std = {}, with shifted Poisson lead time', round(mean, 2), round(std, 2)))
plt.show()

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

print('mean both poisson')
print(np.mean(bayesian_both_poisson))
print(np.mean(ga_both_poisson))
print(np.mean(ols_both_poisson))
print(np.mean(rl_both_poisson))
print(np.mean(ml_both_poisson))
print(np.mean(random_both_poisson))

ax = sns.boxplot(data=[bayesian_both_poisson, ga_both_poisson, ols_both_poisson,
                       rl_both_poisson, ml_both_poisson, random_both_poisson], showfliers = False)
ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Poisson demand over 108 periods,\nmean = {}, with shifted Poisson lead time', round(mean, 2)))
plt.show()

ax = sns.pointplot(data=[bayesian_both_poisson, ga_both_poisson, ols_both_poisson,
                       rl_both_poisson, ml_both_poisson, random_both_poisson],
                   errorbar=("se",2),
                   join = False,
                   capsize=0,
                   markers="_",
                   palette=[colors[label] for label in tick_labels])

ax.set_xticklabels(labels)
ax.set(xlabel='Methods', ylabel='Profit', title=str.format('Normal demand over 108 periods, standard error of 2 std, \n mean = {}, with shifted Poisson lead time', round(mean, 2)))
plt.show()
