import numpy as np
from Item import Item

np.random.seed(123)


class Demandable:
    def __init__(self, holding_cost, fixed_cost, s, S):
        self.inv_level = {}  ## Each item has multiple inv level
        self.inv_pos = {}
        self.upstream = []  ## Each upstream Demandables
        self.holding_cost = (
            holding_cost  ## Possibly change for each item. perhaps a multiplier?
        )

        self.backorder = 0
        self.fixed_cost = fixed_cost
        self.lead_time = 2

        self.arrivals = []
        self.s = s
        self.S = S

    """ def __str__ """

    def change_s(self, new_s):
        self.s = new_s

    def change_S(self, new_S):
        self.S = new_S

    def demand(self, num_demands: int, t) -> None:
        ## For each upstream Demandable, ask for this amount
        for demandable in self.upstream:
            new_items = demandable.get_items(num_demands, t)
            for item in new_items:
                # Increase inv level of items
                self.inv_level[item] += new_items[item]
                # Update inv_pos with new level of items
                self.inv_pos[item] += new_items[item]

    def get_items(self, num_get: int, t):
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
        self.upstream.append(demandable)
        # Change later, perhaps random starting inventory ISSUE here

    def add_item(self, item: "Item", amt: int):
        self.inv_level[item] = amt
        self.inv_pos[item] = amt

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
