from RLDemandable import RLDemandable

class RLRetailer(RLDemandable):
    def __init__(self, name):
        super().__init__(name, 2, 10, 40, 90)
        self.amount_sold = []
        self.selling_price = None
        self.amount_sold_total = 0
        self.profits = []
        
    def set_optimal_selling_price(self, multiplier):
        """Sets optimal price * multipler

        Args:
            multiplier (float): multiplier
        """
        self.selling_price = self.find_optimal_cost() * multiplier
        

    def reset(self, amount = 65):
        """Resets state

        Args:
            amount (int, optional): inventory level. Defaults to 65.
        """
        super().reset(amount)
        self.amount_sold = []
        self.amount_sold_total = 0

    def update_all_inventory(self, t):
        """Updates inv level for all upstream demandables

        Args:
            t (int): timestamp
        """
        #initialise cost to 0 at curr t
        self.amount_sold.append(0) 
        self.costs.append(0)
        self.backorder_costs.append(0)
        self.ordering_costs.append(0)
        self.holding_costs.append(0)
        self.update_inventory(t)
        for demandable in self.upstream:
            demandable.update_all_inventory(t)

    def update_all_demand(self, num_demands: int, t) -> None:
        """Updates inv level and pos for all items for curr and upstream

        Args:
            num_demands (int): amount requested
            t (int): time stamp
        """
        amount_sold = self.update_demand(num_demands)
        self.amount_sold[t] += amount_sold
        self.amount_sold_total += amount_sold
        for item in self.inv_level:
            self.check_s(item, t)
        self.fufill_orders(t)
    
    def update_inventory(self, t):
        """Updates inv level and inv pos

        Args:
            t (int): timestamp
        """
        # self.inv_pos = self.inv_level.copy()
        index = []
        for i in range(len(self.arrivals)):
            arrival = self.arrivals[i]
            time, item, amt = arrival
            if t == time:
                self.inv_level[item] += amt
                index.append(i)
            # self.inv_pos[item] += amt
        self.arrivals = [arrival for i, arrival in enumerate(self.arrivals) if i not in index]

        if self.backorder > 0:
            amt_backordered = min(self.backorder, min(list(self.inv_level.values())))
            for item in self.inv_level:
                self.inv_level[item] -= amt_backordered
            self.backorder -= amt_backordered
            self.amount_sold_total += amt_backordered
            self.amount_sold[t] += amt_backordered

    def calculate_profit(self):
        """calculate profit

        Returns:
            int: profit
        """
        return self.amount_sold_total * self.selling_price - super().get_total_cost()

    def calculate_curr_profit(self, t):
        """calculate profit at time t

        Args:
            t (int): time stamp

        Returns:
            int: profit at time t
        """
        return self.amount_sold[t] * self.selling_price - super().get_curr_total_costs(t)
    
    def __repr__(self):
        return "Retailer({})".format(self.name)