from Demandable import Demandable
import numpy as np
from Item import Item

class Supplier(Demandable):
    def __init__(self, name):
        super().__init__(name, 0, 0, 0, 0)
    
    def add_item_downstream(self, item, amount=65):
        self.add_item(item, np.inf)
        if self.downstream: # Check if list empty
            downstream_demandable = self.downstream[0]
            downstream_demandable.add_item_map(item, self)
            downstream_demandable.add_item_downstream(item, amount)
    
    def update_demand(self, num_get: int):
        return num_get

    def __str__(self):
        return self.name + "\n" + self.print_inv_level() + "\n" + self.print_inv_pos()
    
    def __repr__(self):
        return "Supplier({})".format(self.name)
    
        