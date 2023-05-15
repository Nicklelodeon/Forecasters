import numpy as np

class Demandable:
    def __init__(self, holding_cost):
        self.inv_level = {} ## Each item has multiple inv level
        self.upstream = [] ## Each upstream Demandables
        self.holding_cost = holding_cost ## Possibly change for each item. perhaps a multiplier?
        
    def demand(self, num_demands: int) -> None:
        ## For each upstream Demandable, ask for this amount
        for i in range(len(self.upstream)):
            demandable = self.upstream[i]
            self.inv_level[i] += demandable.getItems(num_demands) ## There is issue here need to think abt implementation

    def get_items(self, num_get: int) -> int:
        items_out = min(num_get, min(self.inv_level))
        for i in range(len(self.inv_level)):
            self.inv_level[i] -= items_out
        return items_out
    
    def add_upstream(self, demandable : "Demandable") -> None:
        self.upstream.append(demandable)
        self.inv_level.append(50000) #Change later, perhaps random starting inventory ISSUE here
    
    def get_hc(self) -> int:
        return sum(self.inv_level) * self.holding_cost
    
    def get_totalhc(self) -> int: #Get all holding cost upstream
        total = self.get_hc()
        for demandable in self.upstream:
            total += demandable.get_totalhc()
        return total