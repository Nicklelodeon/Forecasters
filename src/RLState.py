from RLDemandable import RLDemandable
from RLSupplier import RLSupplier
from RLDistributionCenter import RLDistributionCenter
from RLRetailer import RLRetailer
from RLBasic import RLBasic
from GenerateDemandMonthly import GenerateDemandMonthly
from Stochastic_Lead_Time import Stochastic_Lead_Time
from State import State
import matplotlib.pyplot as plt
import networkx as nx

from Item import Item
import numpy as np
import random
import seaborn as sns 
import pandas as pd

import itertools

class RLState(State):
    def __init__(self):
        self.root = RLBasic(chr(65))
        self.demandables = None
        self.changeable_network = []
        self.network_list = None
        self.demand_list = None
        self.rewards = 0
        self.rewards_list = []
        self.demand = GenerateDemandMonthly()
        self.start_time = 0
        
        self.state = None #network.getcurrstate
        self.curr_time = self.start_time
        
        self.mean = None
        self.std = None
    
    def create_state(self, demandables, mean, std, amount=65, cost=1):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
        self.mean = mean
        self.std = std
        self.demandables = demandables
        network_list = []
        for i in range(len(demandables)):   
            new_demandable = RLBasic(chr(i + 65))
            network_list.append(new_demandable)
        network_list = self.create_network(demandables, network_list)
        
        stl = Stochastic_Lead_Time()
        
        for i in range(len(network_list)):
            network_list[i] = network_list[i].define_demandable()
            network_list[i].add_lead_time(stl)
        network_list = self.create_network(demandables, network_list)

        self.root = network_list[0]
        list_end_upstream = self.root.find_end_upstream()

        for i in range(len(list_end_upstream)):
            end_demandable = list_end_upstream[i]
            item = Item(str(i+1), 1)
            end_demandable.add_item_downstream(item, amount)
        
        self.network_list = network_list
        self.create_changeable_network()
        self.root.set_optimal_selling_price(5)        
        
        for demandable in self.changeable_network:
            demandable.change_s(1000)
        
        # Creates a list of action that the RL model can take
        action_lists = [[40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120] for x in range(len(self.changeable_network))]
        self.action_map = [x for x in itertools.product(*action_lists)]

        self.set_demand_list(self.demand.simulate_normal_no_season(mean=mean, std=std))
        self.reset()
        self.state = self.get_state()
    
    def add_lead_time(self, stochastic_lead_time):
        """Adds different lead time distribution to demandables
        in network

        Args:
            stochastic_lead_time (Lead time distribution): Lead time
        """
        for demandable in self.network_list:
            demandable.add_lead_time(stochastic_lead_time)
    
    def set_demand_list(self, demand_list):
        """Sets demand list

        Args:
            demand_list (list[int]): list of demand
        """
        self.demand_list = demand_list
        
    def step(self, action):
        """State takes one time step

        Args:
            action (int): maps int to action map

        Returns:
            state: state after step
        """
        action_map = self.action_map[action]
        for i in range(len(self.changeable_network)):
            demandable = self.changeable_network[i]
            current_action = action_map[i]
            demandable.change_S(current_action)
        
        self.update_state(self.curr_time)
        self.state = self.get_state()
        reward = self.root.calculate_curr_profit(self.curr_time)
        self.curr_time += 1
        if self.curr_time >= len(self.demand_list):
            done = True
        else:
            done = False
        
        return self.state, reward, done
    
    def get_state(self):
        """gets current state

        Returns:
            State: current state
        """
        lst = []
        for demandable in self.changeable_network:
            lst.extend(demandable.get_state())
        return lst

    def render(self):
        """render visualization (None)
        """
        pass
    
    def reset(self, amount=65):
        """resets state

        Args:
            amount (int, optional): amount level. Defaults to 65.

        Returns:
            state: current state
        """
        for demandable in self.changeable_network:
            demandable.reset(amount)
        self.rewards = 0
        self.rewards_list = []
        self.curr_time = self.start_time
        self.set_demand_list(self.demand.simulate_normal_no_season(mean=self.mean, std=self.std))        
        self.state = self.get_state()
        return self.state 

    def update_state(self, t):
        """Discrete update state

        Args:
            demand (_type_): _description_
            t (int): time
        """
        #self.update_order_point(t)
        self.root.update_all_inventory(t)
        self.root.update_all_demand(self.demand_list[t], t)
        self.root.update_all_cost(t)
        self.rewards += self.root.calculate_curr_profit(t)
        self.rewards_list.append(self.root.calculate_curr_profit(t))

    def calculate_profits(self):
        return self.root.calculate_profit()
           
    def create_network(self, demandables, network):
        """Creates the network of demandables based on demandables list

        Args:
            demandables (list<int>): list of integers s.t list[i] <= list[j] for i <= j and 
            list[-1] == -1 which represents the root (retailer)
            network (list<Demandable>): list of Demandables, len(demandables) == len(network) 

        Returns:
            list<Demandable>: returns list of Demandables with complete connection based
            on demandables list
        """
        for i in range(1,len(demandables)):
            current_demandable = network[demandables[i]]
            current_demandable.add_upstream(network[i])
        return network
    
    def create_changeable_network(self):
        """creates changeable network
        """
        self.changeable_network = self.root.find_changeable_network()

    def print_state(self, t):
        return "time " + str(t) +": \n" + self.root.print_upstream_state()
    

