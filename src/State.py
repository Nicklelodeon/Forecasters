from Demandable import Demandable
from Supplier import Supplier
from DistributionCenter import DistributionCenter
from Retailer import Retailer
from Basic import Basic
from GenerateDemandMonthly import GenerateDemandMonthly
from Stochastic_Lead_Time import Stochastic_Lead_Time


from Item import Item
import numpy as np
import random

class State:
    def __init__(self):
        self.root = Basic(chr(65))
        self.changeable_network = []
        self.demand_list = None
        self.s_S_list = None
        self.rewards = 0
        
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
        
    def set_demand_list(self, demand_list):
        self.demand_list = demand_list
        
    def create_state(self, demandables, amount=65, cost=1):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
        network_list = []
        for i in range(len(demandables)):   
            new_demandable = Basic(chr(i + 65))
            network_list.append(new_demandable)
        network_list = self.create_network(demandables, network_list)
        
        stl = Stochastic_Lead_Time()
        
        for i in range(len(network_list)):
            network_list[i] = network_list[i].define_demandable()
            network_list[i].add_lead_time(stl)
        network_list = self.create_network(demandables, network_list)

        self.root = network_list[0]
        list_end_upstream = self.root.find_end_upstream()

        for end_demandable in list_end_upstream:
            rand_item = Item(str(np.random.randint(1, 1000)), cost)
            end_demandable.add_item_downstream(rand_item, amount)
        
        self.create_changeable_network()
        self.root.set_optimal_selling_price(10)
        

    
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
    
    def valid_check(self, X):
        """Checks the validity of s_S List

        Returns:
            boolean: True if valid else False
        """
        for i in range(len(X)//2):
            index = 2 * i
            if X[index] > X[index + 1]:
                return False
        return True
        

    def total_sum(self):
        """returns cumulative score

        Returns:
           int: sum of all rewards up to this point in time
        """
        return self.rewards


        
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
        self.demand_list = None
        self.s_S_list = None
        self.rewards = 0
    
    def run(self, X):
        for j in range(len(self.changeable_network)):
            small_s = X[2 * j]
            big_S = X[2 * j + 1]
            demandable = self.changeable_network[j]
            demandable.change_order_point(small_s, big_S)
        self.reset()
                
        for i in range(len(self.demand_list)):
            self.update_state(i)
            
        return self.total_sum()
    
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


    def calculate_profits(self):
        return self.root.calculate_profit()
    
    def print_state(self, t):
        return "time " + str(t) +": \n" + self.root.print_upstream_state()
    



