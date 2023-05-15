import numpy as np


class Demandable:
    def __init__(self, holding_cost, fixed_cost, s, S):
        self.inv_level = {}  ## Each item has multiple inv level
        self.upstream = []  ## Each upstream Demandables
        self.holding_cost = (
            holding_cost  ## Possibly change for each item. perhaps a multiplier?
        )
        self.fixed_cost = fixed_cost
        self.lead_time = 2
        self.inv_pos = {}
        self.arrivals = []
        self.s = s
        self.S = S

    def change_s(self, new_s):
        self.s = new_s

    def change_S(self, new_S):
        self.S = new_S

    def demand(self, num_demands: int, t) -> None:
        ## For each upstream Demandable, ask for this amount
        for demandable in upstream:
            new_items = demandable.get_items(num_demands, t)
            for item in new_items:
                # Increase inv level of items
                self.inv_level[item] += new_items[item]
                # Update inv_pos with new level of items
                self.inv_pos[item] += new_items[item]

    def get_items(self, num_get: int, t) -> int:
        items_out = {}
        for i in self.inv_level:
            final_val = self.inv_level - num_get
            final_pos = self.inv_pos - num_get
            if final_val >= 0:
                self.inv_level = final_val
            else:
                self.inv_level = 0
                self.backorder += abs(final_val)
            if final_pos >= 0:
                self.inv_pos = final_val
            else:
                self.inv_level = 0
            if self.inv_level < self.s:
                self.arrivals.append([t + lead_time, self.S - self.inv_level])
            items_out[i] = num_get - self.backorder
        return items_out

    def add_upstream(self, demandable: "Demandable") -> None:
        self.upstream.append(demandable)
        self.inv_level.append(
            50000
        )  # Change later, perhaps random starting inventory ISSUE here

    def get_hc(self) -> int:
        return sum(self.inv_level) * self.holding_cost

    def get_totalhc(self) -> int:  # Get all holding cost upstream
        total = self.get_hc()
        for demandable in self.upstream:
            total += demandable.get_totalhc()
        return total
