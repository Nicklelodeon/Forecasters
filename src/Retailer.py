from Demandable import Demandable

class Retailer(Demandable):
    def __init__(self, name):
        super().__init__(name, 2, 10, 40, 90)
        self.curr_amount_sold = 0
        self.selling_price = None
        self.amount_sold_total = 0
        #self.profits = []
        
    def set_optimal_selling_price(self, multiplier):
        """Sets optimal price * multipler

        Args:
            multiplier (float): multiplier
        """
        self.selling_price = self.find_optimal_cost() * multiplier
        

    def reset(self, amount = 65):
        """Resets state

        Args:
            amount (int, optional): Inventory amount. Defaults to 65.
        """
        super().reset(amount)
        self.curr_amount_sold = 0
        self.amount_sold_total = 0
    
    def clear_previous_time(self):
        """Clears previous time trackers
        """
        self.curr_amount_sold = 0
        super().clear_previous_time()

    def update_all_inventory(self, t):
        """Updates inv level for all upstream demandables

        Args:
            t (int): timestamp
        """
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
        self.curr_amount_sold += amount_sold
        self.amount_sold_total += amount_sold
        for item in self.inv_level:
            self.check_s(item, t)
        self.fufill_orders(t)
    
    def update_inventory(self, t):
        """Updates inv level and inv pos

        Args:
            t (int): timestamp
        """
        index = []
        for i in range(len(self.arrivals)):
            arrival = self.arrivals[i]
            time, item, amt = arrival
            if t == time:
                self.inv_level[item] += amt
                index.append(i)
        self.arrivals = [arrival for i, arrival in enumerate(self.arrivals) if i not in index]

        if self.backorder > 0:
            amt_backordered = min(self.backorder, min(list(self.inv_level.values())))
            for item in self.inv_level:
                self.inv_level[item] -= amt_backordered
            self.backorder -= amt_backordered
            self.amount_sold_total += amt_backordered
            self.curr_amount_sold += amt_backordered

    def calculate_profit(self):
        """Finds total profit made

        Returns:
            int: total profit
        """
        return self.amount_sold_total * self.selling_price - super().get_total_cost()
    
    def calculate_current_profit(self):
        """Get proft at current time

        Returns:
            int: get profit for current time
        """
        profit_sold = self.curr_amount_sold * self.selling_price
        cost_incurred = super().get_total_current_cost()
        return profit_sold - cost_incurred

    def __repr__(self):
        return "Retailer({})".format(self.name)
