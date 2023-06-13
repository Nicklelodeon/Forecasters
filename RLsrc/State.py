from Demandable import Demandable
from Supplier import Supplier
from DistributionCenter import DistributionCenter
from Retailer import Retailer
from Basic import Basic
from GenerateDemandMonthly import GenerateDemandMonthly
from Stochastic_Lead_Time import Stochastic_Lead_Time
import matplotlib.pyplot as plt
import networkx as nx

from Item import Item
import numpy as np
import random
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from gym import Env
from gym.spaces import MultiDiscrete

class State(Env):
    def __init__(self):
        self.root = Basic(chr(65))
        self.demandables = None
        self.changeable_network = []
        self.network_list = None
        self.demand_list = None
        self.rewards = 0
        self.rewards_list = []
        self.demand = GenerateDemandMonthly()
        
        ### RL attributes ###
        ### To be edited ###
        self.action_space = MultiDiscrete([150,150])
        
        ##[Inventory pos 1, Inventory lev 1, Inventory pos 2, Inventory lev 2]
        self.observation_space = MultiDiscrete([200,200,200,200])
        
        self.state = None #network.getcurrstate
        self.curr_time = 0
    
    def create_state(self, demandables, amount=65, cost=1):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
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

        for end_demandable in list_end_upstream:
            rand_item = Item(str(np.random.randint(1, 1000)), cost)
            end_demandable.add_item_downstream(rand_item, amount)
        
        self.network_list = network_list
        self.create_changeable_network()
        self.root.set_optimal_selling_price(1.5)
        self.set_demand_list(self.demand.simulate_normal_no_season())
        self.update_state(self.curr_time)
        self.state = self.root.get_state()
    
    def set_demand_list(self, demand_list):
        self.demand_list = demand_list
        
    def step(self, action):
        # Action eg [20, 10]
        ##### self.root.order_item(1, 20)
        ##### self.root.oder_item(2, 30)
        
        # Calculate reward vary this before step and after step
        reward = self.root.calculate_curr_profit(self.curr_time)
        # print("#################### REWARD: ", reward)
        
        for i in range(len(action)):
            self.root.order_item(i, action[i], self.curr_time)

        # Get the current state from root
        self.state = self.root.get_state()
        
        # Prints state
        print(self.print_state(self.curr_time))
        print(self.state)
        
        # Update time
        self.curr_time += 1
        
        # Updates state at t+1, after action is taken
        # Checks if time exceeds the demand list
        if self.curr_time >= len(self.demand_list):
            done = True
        else:
            done = False
            self.update_state(self.curr_time)


       
        # Calculate reward
        # reward = self.root.calculate_profit(self.curr_time)
        
        # Check if time exceeds the demand list
        """ if self.curr_time >= len(self.demand_list):
            done = True
        else:
            done = False """
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        """RL function reset
        """
        amount = 65
        for demandable in self.changeable_network:
            demandable.reset(amount)
        self.rewards = 0
        self.rewards_list = []
        self.curr_time = 0
        self.set_demand_list(self.demand.simulate_normal_no_season())
        self.update_state(self.curr_time)
        self.state = self.root.get_state()
        return self.state

    def update_state(self, t):
        """Discrete update state

        Args:
            demand (_type_): _description_
            t (int): time
        """
        #self.update_order_point(t)
        self.root.update_all_inventory(t)
        self.root.update_demand(self.demand_list[t])
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
        self.changeable_network = self.root.find_changeable_network()
            
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

    def print_state(self, t):
        return "time " + str(t) +": \n" + self.root.print_upstream_state()
    
    def plot_rewards(self):
        df = pd.DataFrame(columns=["time", "rewards"])
        for i, val in enumerate(self.rewards_list):
            df.loc[len(df.index)] = [i, val]
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.pointplot(data=df, x='time', y='rewards', ax=ax)
        plt.show()
    



