from Demandable import Demandable

class Retailer(Demandable):
    def __init__(self, name, selling_price):
        super().__init__(name, 5, 10, 10, 40, 90)
        self.amount_sold = []
        self.selling_price = selling_price

    def reset(self):
        self.inv_level  = dict.fromkeys(self.inv_level, 65)
        self.inv_pos = dict.fromkeys(self.inv_pos, 65)
        self.ordering_costs = []
        self.holding_costs = []
        self.backorder_costs = []
        self.backorder = 0
        self.costs = []
        self.arrivals = []
        self.amount_sold = []

    def update_all_demand(self, num_demands: int, t) -> None:
        """Updates inv level and pos for all items for curr and upstream

        Args:
            num_demands (int): amount requested
            t (int): time stamp
        """
        amount_sold = self.update_demand(num_demands)
        self.amount_sold.append(amount_sold)
        for item in self.inv_level:
            self.check_s(item, t)
    
    def calculate_profit(self, t):
        return self.amount_sold[t] * self.selling_price - super().get_total_cost(t)
    
    def __repr__(self):
        return "Retailer({})".format(self.name)




    

        