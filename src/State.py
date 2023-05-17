from Demandable import Demandable
from Item import Item
import numpy as np

class State:

    def __init__(self, root):
        self.root = root

    def create_state(self, demandables):
        """create state

        Args:
            demandables (list<int>): list of integers
        """
        head = self.root
        network_list = [head]
        for i in range(1, len(demandables)):
            new_demandable = Demandable(20, 100, 50 ,100)
            network_list.append(new_demandable)
            current_demandable = network_list[demandables[i]]
            current_demandable.add_upstream(new_demandable)
        list_end_upstream = head.find_end_upstream()
        
        for end_demandable in list_end_upstream:
            rand_item = Item(str(np.random.randint(1, 1000)), 10)
            end_demandable.add_item_downstream(rand_item)

    def update_state(self, demand, t):
        self.root.update_all_inventory(i)
        self.root.update_all_demand(demand, t)
        self.root.update_all_cost(t)
    
    def print_state(self, t):
        return helper_print(self.root, "t: ")

    def helper_print(self, root, string):
        string += str(root)
        for demandable in root.upstream:
            return helper_print(demandable, string, t)


    

    



