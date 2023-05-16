import numpy as np
from Item import Item

np.random.seed(123)


class Demandable:
    def __init__(self, holding_cost, fixed_cost, s, S):
        self.inv_level = {}  ## Each item has multiple inv level
        self.inv_pos = {}
        self.upstream = []  ## Each upstream Demandables
        self.downstream = [] ## Each downstream Demandables
        self.holding_cost = holding_cost  ## Possibly change for each item. perhaps a multiplier

        self.backorder = 0
        self.fixed_cost = fixed_cost
        self.lead_time = 2

        self.arrivals = []
        self.s = s
        self.S = S
        

    
    
    

    def change_s(self, new_s):
        """Changes lower bound s

        Args:
            new_s (int): new s 
        """
        self.s = new_s

    def change_S(self, new_S):
        """Changes upper bound s

        Args:
            new_S (int): new S
        """
        self.S = new_S


    def demand(self, num_demands: int, t) -> None:
        """Request stock from upstream

        Args:
            num_demands (int): amount requested
            t (int): time stamp
        """
        ## For each upstream Demandable, ask for this amount
        for demandable in self.upstream:
            new_items = demandable.get_items(num_demands)
            for item in new_items:
                # Increase inv level of items
                self.inv_level[item] += new_items[item]
                # Update inv_pos with new level of items
                self.inv_pos[item] += new_items[item]

    def get_items(self, num_get: int):
        """Update inv level and inv pos for upstream

        Args:
            num_get (int): amount requested from downstream

        Returns:
            items_out (int): amount that each upstream demandable can provide 
        """
        items_out = {}
        min_item = min(min(list(self.inv_level.values())), num_get)
        curr_backorder = num_get - min_item
        self.backorder += curr_backorder
        # if min_item >= num_get: # The amount requested is less than min
        for item in self.inv_level:
            items_out[item] = min_item
            self.inv_level[item] -= min_item
        return items_out

    def add_upstream(self, demandable: "Demandable") -> None:
        """Adds a demandable into upstream

        Args:
            demandable (Demandable): To be added in upstream
        """
        self.upstream.append(demandable)
        demandable.add_downstream(self)
        # Change later, perhaps random starting inventory ISSUE here

        
    def add_downstream(self, demandable: "Demandable") -> None:
        self.downstream.append(demandable)
    
    #Add items downstream with random amount
    def add_item_downstream(self, item: "Item"):
        self.add_item(item, np.random.randint(4000, 7000))
        if self.downstream: # Check if list empty
            self.downstream[0].add_item_downstream(item)
    
    def find_end_upstream(self) -> list:
        leaves = []
        if self.upstream: #upstream not empty
            for demandable in self.upstream:
                leaves += demandable.find_end_upstream()
        else:
            leaves += [self]
        return leaves
            

    def add_item(self, item: "Item", amt = 0):
        """Add item to demandable and its downstream, create inv level and inv pos

        Args:
            item (Item): Item added
            amt (int): Amount of item to be added
        """
        self.inv_level[item] = amt
        self.inv_pos[item] = amt

    def order_cost(self, item, amt, t):
        """Adds order and returns cost of ordering

        Args:
            item (Item): item to be ordered
            amt (int): amount to order
            t (int): timestamp

        Returns:
            int: total order cost
        """
        self.arrivals.append([t + lead_time, item, amt])
        return amt * item.get_cost + self.fixed_cost

    def update_inventory(self, t):
        self.inv_pos = self.inv_level.copy()
        for arrival in self.arrivals:
            time, item, amt = arrival
            if t == time:
                self.inv_level[item] += amt
                arrivals.remove(arrival)
            self.inv_pos[item] += amt

    def get_hc(self) -> int:
        total = 0
        for item in self.inv_level:
            item_cost = item.get_cost()
            item_amt = self.inv_level[item]
            total += item_cost * item_amt
        return total * self.holding_cost

    def get_totalhc(self) -> int:  # Get all holding cost upstream
        total = self.get_hc()
        for demandable in self.upstream:
            total += demandable.get_totalhc()
        return total
