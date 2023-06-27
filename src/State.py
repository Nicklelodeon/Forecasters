from Demandable import Demandable
from Supplier import Supplier
from DistributionCenter import DistributionCenter
from Retailer import Retailer
from Basic import Basic
from GenerateDemandMonthly import GenerateDemandMonthly
from Stochastic_Lead_Time import Stochastic_Lead_Time
import matplotlib.pyplot as plt
import networkx as nx
from concurrent import futures
from multiprocessing import Pool, freeze_support, cpu_count
import os

from Item import Item
import numpy as np
import random
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd


class State:
    def __init__(self):
        self.root = Basic(chr(65))
        self.demandables = None
        self.changeable_network = []
        self.network_list = None
        self.demand_list = None
        self.s_S_list = None
        self.rewards = 0
        self.rewards_list = []
        self.start_inventory = 65
        
        ### User input
        self.periods = None
        self.iterations = None
        self.demand_generator = GenerateDemandMonthly()
        self.demand_matrix = None
        self.mean = None
        self.std = None
        
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
    
    
    def set_demand_matrix(self, demand_matrix):
        self.demand_matrix = demand_matrix
        
    def create_state(self, demandables, amount=65, period=108, iterations=500, mean=5, std=2):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
        self.periods = period
        self.iterations = iterations
        self.mean = mean
        self.std = std
        np.random.seed(1234) # set same demand matrix
        self.demand_matrix = np.reshape(self.demand_generator.simulate_normal_no_season(\
            periods = self.periods * self.iterations, mean=self.mean, std=self.std),\
                (self.iterations, self.periods))
        
        self.demandables = demandables
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
        
        for i in range(len(list_end_upstream)):
            end_demandable = list_end_upstream[i]
            item = Item(str(i+1), 1)
            end_demandable.add_item_downstream(item, self.start_inventory)
        
        self.network_list = network_list
        self.create_changeable_network()
        self.root.set_optimal_selling_price(5)
        print([x for x in self.demand_generator.simulate_normal_no_season(\
            periods = self.periods * self.iterations, mean=self.mean, std=self.std) if x < 0])
        print("mean:", mean, "std:", std)
        self.demand_generator.visualize()


    def create_normal_demand(self, period = 108, iterations = 500, mean = 5, std = 2): 
        return np.reshape(self.demand_generator.simulate_normal_no_season(\
            periods = period * iterations, mean=mean, std=std),\
                (iterations, period))
    
    def create_poisson_demand(self, period = 108, iterations = 500, mean = 5): 
        return np.reshape(self.demand_generator.simulate_poisson_no_season(\
            periods = period * iterations, mean=mean),\
                (iterations, period))
        
    def run(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1): #6 params
        if (s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1):
            return -100000
        self.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
        self.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
        self.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
        total_sum = 0
        np.random.seed(5678)
        for z in range(self.iterations):
            self.reset(self.start_inventory)
            self.set_demand_list(self.demand_matrix[z])
            for i in range(self.periods):
                self.update_state(i)
            total_sum += self.calculate_profits()
        return total_sum / self.iterations


    def test_no_season(self, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
        self.changeable_network[0].change_order_point(round(s_r1), round(S_r1))
        self.changeable_network[1].change_order_point(round(s_DC1), round(S_DC1))
        self.changeable_network[2].change_order_point(round(s_DC2), round(S_DC2))
        np.random.seed(7890) # set same demand matrix
        self.demand_matrix = np.reshape(self.demand_generator.simulate_normal_no_season(\
            periods = self.periods * self.iterations, mean=self.mean, std=self.std),\
                (self.iterations, self.periods))
        lst = []
        np.random.seed(2340)
        for z in range(self.iterations):
            self.reset(self.start_inventory)
            self.set_demand_list(self.demand_matrix[z])
            for i in range(self.periods):
                self.update_state(i)
            lst.append(self.calculate_profits())
        print("mean:", np.mean(lst))
        print("std:", np.std(lst, ddof=1)/(self.iterations)**0.5)
                    
    def show_network(self):
        """Creates a tree graph of the supply chain system
        """
        def find_points(i):
            lst = []
            x = i - 1
            interval = 2 * x
            small_interval = interval/(i+1)
            for i in range(1, i+1):
                lst.append(x - (i * small_interval))
            return lst
                
        adj_lst = []
        demandables_list = ["A"]
        for i in range(1, len(self.demandables)):
            demandables_list.append(chr(i + 65))
            head = self.demandables[i]
            adj_lst.append((chr(head + 65), chr(i + 65)))
        x_pos = [0] * len(self.demandables)
        
        for i in range(1, len(self.demandables)):
            head = self.demandables[i]
            x_pos[i] = x_pos[head] + 1
        depth_count = [0] * (max(x_pos) + 1)
        
        for i in x_pos:
            depth_count[i] += 1
        lst = list(map(lambda x, y: [x, y], demandables_list, x_pos ))
        dic = {}
        
        for i in range(len(depth_count)):
            dic[i] = find_points(depth_count[i])
            
        for i in range(len(lst)):
            curr_list = lst[i]
            curr_depth = curr_list[1]
            get_y = dic[curr_depth].pop()
            curr_list.append(get_y)
        
        dic2 = {}
        for demandable in self.network_list:
            name = demandable.name
            if isinstance(demandable, Retailer):
                dic2[name] = "Retailer: \n" + name
            elif isinstance(demandable, DistributionCenter):
                dic2[name] = "DC: \n" + name
            else:
                dic2[name] = "Supplier: \n" + name
        
        for i in range(len(adj_lst)):
            pos1 = adj_lst[i][0]
            pos2 = adj_lst[i][1]
            adj_lst[i] = (dic2[pos1], dic2[pos2])
            
        G = nx.DiGraph()
        for i in lst:
            G.add_node(dic2[i[0]], pos=(i[1], i[2]))
        G.add_edges_from(adj_lst)
        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_size=750, node_color='lightblue', font_size=12, font_weight='bold', width=2,
        arrowstyle='<-', arrowsize=15)
        plt.axis('equal')
        plt.show()
        
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
            
    def reset(self, amount=65):
        """Resets state
        """
        for demandable in self.changeable_network:
            demandable.reset(amount)
        self.demand_list = None
        self.s_S_list = None
        self.rewards = 0
    
    """ def run(self, X):
        for j in range(len(self.changeable_network)):
            small_s = X[2 * j]
            big_S = X[2 * j + 1]
            demandable = self.changeable_network[j]
            demandable.change_order_point(small_s, big_S)
        self.reset()
                
        for i in range(len(self.demand_list)):
            self.update_state(i)
            
        return self.total_sum() """
    
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
        # self.rewards += self.root.calculate_curr_profit(t)
        # self.rewards_list.append(self.root.calculate_curr_profit(t))
        


    def calculate_profits(self):
        return self.root.calculate_profit()
    
    def print_state(self, t):
        return "time " + str(t) +": \n" + self.root.print_upstream_state()
    
    def plot_rewards(self):
        df = pd.DataFrame(columns=["time", "rewards"])
        for i, val in enumerate(self.rewards_list):
            df.loc[len(df.index)] = [i, val]
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.pointplot(data=df, x='time', y='rewards', ax=ax)
        # label points on the plot
        # for x, y in zip(df['time'], df['cost']):
        #     plt.text(x = x, y = y+10, s = "{:.0f}".format(y), color = "purple") 
        # # sns.relplot(kind='line', data=df, x='time', y='cost', hue='type')
        plt.show()
    


