from Demandable import Demandable
from Supplier import Supplier
from DistributionCenter import DistributionCenter
from Retailer import Retailer
from Basic import Basic

from Item import Item
import numpy as np

class State:
    def __init__(self):
        self.root = Basic(chr(65))
        
    def create_network(self, demandables, network):
        for i in range(1,len(demandables)):
            current_demandable = network[demandables[i]]
            current_demandable.add_upstream(network[i])
        return network
            
        
    def create_state(self, demandables):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
        #head = self.root
        network_list = []
        for i in range(len(demandables)):   
            new_demandable = Basic(chr(i + 65))
            network_list.append(new_demandable)
        network_list = self.create_network(demandables, network_list)
        
        for i in range(len(network_list)):
            network_list[i] = network_list[i].define_demandable()
        network_list = self.create_network(demandables, network_list)
        #print("HERE AT LINE 37 PRINT 2",network_list)

     
        self.root = network_list[0]
        #print("ROOT", str(self.root))
        #print("PRINT 3", head.upstream)
        list_end_upstream = self.root.find_end_upstream()
        #print("LINE 46", list_end_upstream)
        #print("LINE 45")
        
        
        for end_demandable in list_end_upstream:
            rand_item = Item(str(np.random.randint(1, 1000)), 10)
            end_demandable.add_item_downstream(rand_item)
        #print("END DEMANDABLE",list_end_upstream[0].inv_level, list_end_upstream[0].downstream[0].inv_level)
        
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



