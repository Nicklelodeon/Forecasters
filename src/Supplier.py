from Demandable import Demandable
import numpy as np
from Item import Item

class Supplier(Demandable):
    def __init__(self, name):
        super().__init__(name, 0, 0, 0, 0, 0)
    
    def add_item_downstream(self, item: "Item"):
        self.add_item(item, np.inf)
        if self.downstream: # Check if list empty
            downstream_demandable = self.downstream[0]
            downstream_demandable.add_item_map(item, self)
            downstream_demandable.add_item_downstream(item)
    
    def update_demand(self, num_get: int):
        return num_get
        
    
        