from State import State 
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
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd

class MLState(State):
    def __init__(self):
        super().__init__()
    
    def change_demand(self, demand_list):
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