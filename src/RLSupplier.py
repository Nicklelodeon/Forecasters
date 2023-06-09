from RLDemandable import RLDemandable
import numpy as np
from Item import Item

class RLSupplier(RLDemandable):
    def __init__(self, name):
        super().__init__(name, 0, 0, 0, 0)
    
    def add_item_downstream(self, item, amount=65):
        """adds specified amount of item to each downstream demandable

        Args:
            item (Item)
            amount (int, optional): amount of an item to be added. Defaults to 65.
        """
        self.add_item(item, np.inf)
        if self.downstream: # Check if list empty
            downstream_demandable = self.downstream[0]
            downstream_demandable.add_item_map(item, self)
            downstream_demandable.add_item_downstream(item, amount)
    
    def update_demand(self, num_get: int):
        return num_get

    def update_all_cost(self, t):
        return 
    
    def __repr__(self):
        return "Supplier({})".format(self.name)
    
        