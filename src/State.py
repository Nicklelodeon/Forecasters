from Demandable import Demandable
from Supplier import Supplier
from DistributionCenter import DistributionCenter
from Retailer import Retailer
from Basic import Basic
from Monthly_Interval import GenerateDemandMonthly

from Item import Item
import numpy as np
import random

class State:
    def __init__(self):
        self.root = Basic(chr(65))
        self.changeable_network = []
        self.demand_class = GenerateDemandMonthly()
        self.demand_class.simulate(1)
        self.demand_list = self.demand_class.get_demand()
        self.s_S_list = None
        self.rewards = []
        
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
        self.changeable_network = self.root.find_changeable_network()
        
        
    def create_state(self, demandables):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
        network_list = []
        for i in range(len(demandables)):   
            new_demandable = Basic(chr(i + 65))
            network_list.append(new_demandable)
        network_list = self.create_network(demandables, network_list)
        
        for i in range(len(network_list)):
            network_list[i] = network_list[i].define_demandable()
        network_list = self.create_network(demandables, network_list)

        self.root = network_list[0]
        list_end_upstream = self.root.find_end_upstream()

        for end_demandable in list_end_upstream:
            rand_item = Item(str(np.random.randint(1, 1000)), 10)
            end_demandable.add_item_downstream(rand_item)
        
        self.create_changeable_network()
        
    def take_vector(self, array):
        """Assign the array to the s_S_list

        Args:
            array (list<int>): list of integers
        """
        self.s_S_list = array

    def create_array(self, s_min, s_max, S_min, S_max):
        arr = []
        for x in range(len(self.changeable_network)):
            s = random.sample([x for x in range(s_min, s_max + 1)], 12)
            S = random.sample([x for x in range(S_min, S_max + 1)], 12)
            for i in range(12):
                arr.append(s[i])
                arr.append(S[i])
        self.take_vector(arr)
    
    def score(self, t):
        """returns score
        """
        return self.rewards[t]
    
    def valid_check(self):
        """Checks the validity of s_S List

        Returns:
            boolean: True if valid else False
        """
        for i in range(len(self.s_S_list)//2):
            index = 2 * i
            if self.s_S_list[index] > self.s_S_list[index + 1]:
                return False
        return True
        

    def total_sum(self):
        """returns cumulative score

        Returns:
           int: sum of all rewards up to this point in time
        """
        self.valid_check()
        return sum(self.rewards)
        
    def print_network(self):
        """Debugging function to print Demandables in network
        """
        print(self.root.print_upstream())
        
    def update_order_point(self, t):
        """Changes small and big s and S

        Args:
            t (int): time
        """
        for i in range(len(self.changeable_network)):
            demandable = self.changeable_network[i]
            point = i * 24 + (2 * t)
            small_s = self.s_S_list[point]
            big_S = self.s_S_list[point + 1]
            demandable.change_order_point(small_s, big_S)
            
    def reset(self):
        """Resets state
        """
        for demandable in self.changeable_network:
            demandable.reset()
        self.s_S_list = None
        self.rewards = []
        
        
    def update_state(self, t):
        """Discrete update state

        Args:
            demand (_type_): _description_
            t (int): time
        """
        self.update_order_point(t)
        self.root.update_all_inventory(t)
        self.root.update_all_demand(self.demand_list[t], t)
        self.root.update_all_cost(t)
        self.rewards.append(self.root.calculate_profit(t))

    
    def print_state(self, t):
        return "time " + str(t) +": \n" + self.root.print_upstream_state()
    



