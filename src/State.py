from Demandable import Demandable
from Supplier import Supplier
from DistributionCenter import DistributionCenter
from Retailer import Retailer
from Basic import Basic
from Monthly_Interval import GenerateDemandMonthly

from Item import Item
import numpy as np

class State:
    def __init__(self):
        self.root = Basic(chr(65))
        self.changable_network = []
        self.demand_class = GenerateDemandMonthly()
        self.demand_class.simulate(1)
        self.demand_list = self.demand_class.get_demand()
        
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
        
    def print_network(self):
        """Debugging function to print Demandables in network
        """
        print(self.root.print_upstream())
            
    def update_state(self, demand, t):
        self.root.update_all_inventory(t)
        self.root.update_all_demand(demand, t)
        self.root.update_all_cost(t)
    
    def print_state(self, t):
        return "time " + str(t) +": \n" + self.root.print_upstream_state()
    
    """ self.helper_print(self.root, "time " + str(t) +": \n")


    def helper_print(self, root, string):
        string += str(root)
        for demandable in root.upstream:
            return string + self.helper_print(demandable, string)
        return string """



