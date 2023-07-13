import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from RLState import RLState
from GenerateDemandMonthly import GenerateDemandMonthly

from Poisson_Stochastic_Lead_Time import Poisson_Stochastic_Lead_Time
from Stochastic_Lead_Time import Stochastic_Lead_Time

from PPO import PPO

# Read Data
df = pd.read_csv('./src/TOTALSA.csv')
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

real_data = df['TOTALSA'].round().tolist()


env = RLState()
env.create_state([-1, 0, 1, 1, 2, 2], mean=mean, std=std)
dimension = len(env.action_map)

# Inputs for PPO
has_continuous_action_space = False # continuous action space; else discrete
action_std = 0.6            # starting std for action distribution (Multivariate Normal)

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
K_epochs = 20
lr_actor = 0.00005      # learning rate for actor network
lr_critic = 0.0001       # learning rate for critic network

random_seed = 1234         # set random seed if required (0 = no random seed)

state_dim = 9
action_dim = dimension

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Load model
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent.policy_old.load_state_dict(torch.load(map_location=torch.device('cpu'),f="./src/RLmodel.pt"))
ppo_agent.policy.load_state_dict(torch.load(map_location=torch.device('cpu'),f="./src/RLmodel.pt"))

demand_generator = GenerateDemandMonthly()




def test_no_season():
    """Test RL model on 108 period normal demand

    Returns:
        list[int]: results in a list
    """
    period = 108
    iterations = 500
    stl = Stochastic_Lead_Time()
    env.add_lead_time(stl)

    np.random.seed(7890)
    demand_matrix = np.reshape(demand_generator.simulate_normal_no_season(\
            periods = period * iterations, mean=mean, std=std),\
                (iterations, period))
    
    reward_RL = []
    for demand_list in demand_matrix:
        reward_total = 0
        state = env.reset()
        env.set_demand_list(demand_list)
        done = False
        reward_sub = 0
        
        while not done:
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            reward_sub += reward
            if done:
                break
        reward_total += reward_sub
        reward_RL.append(reward_sub)
    return reward_RL

def test_no_season_24_period():
    """Test RL model on 24 period normal demand

    Returns:
        list[int]: results in a list
    """
    period = 24
    iterations = 500
    stl = Stochastic_Lead_Time()
    env.add_lead_time(stl)
    np.random.seed(1357)
    demand_matrix = np.reshape(demand_generator.simulate_normal_no_season(\
            periods = period * iterations, mean=mean, std=std),\
                (iterations, period))
    
    reward_RL = []
    for demand_list in demand_matrix:
        reward_total = 0
        state = env.reset()
        env.set_demand_list(demand_list)
        done = False
        reward_sub = 0
        
        for i in range(period):
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            reward_sub += reward
            if done:
                break
        reward_total += reward_sub
        reward_RL.append(reward_sub)
    return reward_RL

def test_poisson_no_season():
    """Test RL model on 108 period poisson demand

    Returns:
        list[int]: results in a list
    """
    period = 108
    iterations = 500
    stl = Stochastic_Lead_Time()
    env.add_lead_time(stl)
    np.random.seed(12340)
    demand_matrix = np.reshape(demand_generator.simulate_poisson_no_season(\
            periods = period * iterations, mean=mean),\
                (iterations, period))
    
    reward_RL = []
    for demand_list in demand_matrix:
        reward_total = 0
        state = env.reset()
        env.set_demand_list(demand_list)
        done = False
        reward_sub = 0
        
        while not done:
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            reward_sub += reward
            if done:
                break
        reward_total += reward_sub
        reward_RL.append(reward_sub)
    return reward_RL

def test_no_season_poisson_lead_time():
    """Test RL model on 108 period normal demand
    and shifted poisson lead time

    Returns:
        list[int]: results in a list
    """
    period = 108
    iterations = 500
    stl = Poisson_Stochastic_Lead_Time()
    env.add_lead_time(stl)

    np.random.seed(9988) # set same demand matrix
    demand_matrix = np.reshape(demand_generator.simulate_normal_no_season(\
            periods = period * iterations, mean=mean, std=std),\
                (iterations, period))
    
    reward_RL = []
    for demand_list in demand_matrix:
        reward_total = 0
        state = env.reset()
        env.set_demand_list(demand_list)
        done = False
        reward_sub = 0
        
        while not done:
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            reward_sub += reward
            if done:
                break
        reward_total += reward_sub
        reward_RL.append(reward_sub)
    return reward_RL

def test_poisson_no_season_poisson_lead_time():
    """Test RL model on 108 period poisson demand
    and shifted poisson lead time

    Returns:
        list[int]: results in a list
    """
    period = 108
    iterations = 500
    stl = Poisson_Stochastic_Lead_Time()
    env.add_lead_time(stl)

    np.random.seed(24865) # set same demand matrix
    demand_matrix = np.reshape(demand_generator.simulate_poisson_no_season(\
            periods = period * iterations, mean=mean),\
                (iterations, period))
    
    reward_RL = []
    for demand_list in demand_matrix:
        reward_total = 0
        state = env.reset()
        env.set_demand_list(demand_list)
        done = False
        reward_sub = 0
        
        while not done:
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            reward_sub += reward
            if done:
                break
        reward_total += reward_sub
        reward_RL.append(reward_sub)
    return reward_RL

def test_real_data():
    """Test RL model on real data

    Returns:
        int: result
    """
    stl = Stochastic_Lead_Time()
    env.add_lead_time(stl)
    total_sum = 0
    state = env.reset()
    done = False
    
    env.set_demand_list(real_data)
    while not done:
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            total_sum += reward
            if done:
                break
    return total_sum


# a = test_no_season()
# # np.savetxt("./src/RL_no_season.csv", a)

# b = test_no_season_24_period()
# # np.savetxt("./src/RL_no_season_24.csv", b)

# c = test_poisson_no_season()
# # np.savetxt("./src/RL_poisson_no_season.csv", c)

# d = test_no_season_poisson_lead_time()
# # np.savetxt("./src/RL_no_season_poisson_lead_time.csv", d)

# e = test_poisson_no_season_poisson_lead_time()
# np.savetxt("./src/RL_poisson_no_season_poisson_lead_time.csv", e)
# print("test", test_real_data())
# f = np.array(test_real_data())
# print(f)
# np.savetxt("./src/RL_real_data.csv", f)




# df = pd.DataFrame({"RL_no_season": a,
#                    "RL_no_season_24" : b,
#                    "RL_poisson_no_season": c,
#                    "RL_no_season_poisson_lead_time": d,
#                    "RL_poisson_no_season_poisson_lead_time":e})

# df.to_csv("./src/RL_Dataframe.csv")
