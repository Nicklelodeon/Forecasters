import numpy as np
from Item import Item

np.random.seed(123)


class Demandable:
    def __init__(self, holding_cost, fixed_cost, s, S):
        self.inv_level = {}  ## Each item has multiple inv level
        self.inv_pos = {}
        self.inv_map = {} ## Has the inventory to Demandable
        
        self.upstream = []  ## Each upstream Demandables
        self.downstream = [] ## Each downstream Demandables
        self.holding_cost = holding_cost  ## Possibly change for each item. perhaps a multiplier

        self.backorder = 0
        self.fixed_cost = fixed_cost
        self.lead_time = 2
        self.costs = []
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



    def update_all_demand(self, num_demands: int, t) -> None:
        """Request stock from upstream

        Args:
            num_demands (int): amount requested
            t (int): time stamp
        """
        ## For each upstream Demandable, ask for this amount
        
        for demandable in self.upstream:
            new_items = demandable.update_demand(num_demands)
            for item in new_items:
                # Increase inv level of items
                self.arrivals.append([t + self.lead_time, item, new_items[item]])

    def update_demand(self, num_get: int):
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
        """Adds a demandable into downstream, called after
        add_upstream function

        Args:
            demandable (Demandable): demandable
        """
        self.downstream.append(demandable)
    
    def add_item_map(self, item, Demandable):
        """Maps item to Demandable in self.inv_map

        Args:
            item (Item): An Item
            Demandable (Demandable): Direct upstream Demandable
        """
        self.inv_map[item] = Demandable
        
    #Add items downstream with random amount
    def add_item_downstream(self, item: "Item"):
        """Adds items to all the downstream

        Args:
            item (Item): Item added
        """
        
        self.add_item(item, np.random.randint(4000, 7000))
        if self.downstream: # Check if list empty
            downstream_demandable = self.downstream[0]
            downstream_demandable.add_item_map(item, self)
            downstream_demandable.add_item_downstream(item)            
    
    def find_end_upstream(self) -> list:
        """Finds the topmost upstream demandable

        Returns:
            list of demandables 
        """
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

    def update_order(self, item, amt, t):
        """Adds order and updates cost of ordering

        Args:
            item (Item): item to be ordered
            amt (int): amount to order
            t (int): timestamp

        """
        self.update_all_demand(amt, t)
        ordering_cost = amt * item.get_cost() + self.fixed_cost
        self.costs.append(ordering_cost)
        
    def update_all_order(self, t):
        """Orders for curr demandable and all upstream demandable

        Args:
            t (int): timestamp
        """
        self.check_s(t)
        for demandable in self.upstream:
            demandable.update_all_order(t)


    def check_s(self, t):
        """Orders if curr inv level < s

        Args:
            item (Item): item to be ordered
            amt (int): amount to order
            t (int): timestamp
        """
        for item in self.inv_level:
            if self.inv_pos[item] < self.s:
                self.update_order(item, self.S - self.inv_pos[item], t)


    def update_inventory(self, t):
        """Updates inv level and inv pos

        Args:
            t (int): timestamp
        """
        self.inv_pos = self.inv_level.copy()
        for arrival in self.arrivals:
            time, item, amt = arrival
            if t == time:
                self.inv_level[item] += amt
                self.arrivals.remove(arrival)
            self.inv_pos[item] += amt

    def update_all_inventory(self, t):
        """Updates inv level for all upstream demandables

        Args:
            t (int): timestamp
        """
        self.update_inventory(t)
        for demandable in self.upstream:
            demandable.update_all_inventory(t)
 
    def get_hc(self) -> int:
        """Returns holding cost for current demandable

        Returns:
            int: holding cost for current demandable
        """
        total = 0
        for item in self.inv_level:
            item_cost = item.get_cost()
            item_amt = self.inv_level[item]
            total += item_cost * item_amt
        return total * self.holding_cost

    def get_total_cost(self, t) -> int: 
        """Returns total cost for all upstream demandable

        Args:
            t (int): timestamp

        Returns:
            int: total cost incurred by all upstream demandable
        """
        total = self.get_curr_cost(t)
        for demandable in self.upstream:
            total += demandable.get_total_cost(t)
        return total

    def get_curr_cost(self, t):
        """Retrieves total cost at specified time stamp for curr demandable

        Args:
            t (int): time stamp

        Returns:
            int: cost at specified time stamp
        """
        return self.costs[t]

    def update_all_cost(self, t):
        """Add hc into total cost

        Args:
            t (int): curr timestamp
        """
        if (len(self.costs) == t + 1):
            self.costs[t] += self.get_hc()
        else:
            self.costs.append(self.get_hc())
        for demandable in self.upstream:
            demandable.update_all_cost(t)

    