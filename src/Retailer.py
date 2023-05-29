from Demandable import Demandable

class Retailer(Demandable):
    def __init__(self, name, selling_price):
        super().__init__(name, 2, 15, 40, 90)
        #super().__init__(name, 1, 2, 3, 40, 90)
        self.amount_sold = []
        self.selling_price = selling_price
        self.amount_sold_total = 0
        
    def set_optimal_selling_price(self, multiplier):
        """Sets optimal price * multipler

        Args:
            multiplier (float): multiplier
        """
        print("Optimal",self.find_optimal_cost())
        self.selling_price = self.find_optimal_cost() * multiplier

    def reset(self, amount = 65):
        """ self.inv_level  = dict.fromkeys(self.inv_level, amount)
        self.inv_pos = dict.fromkeys(self.inv_pos, amount)
        self.ordering_costs = []
        self.holding_costs = []
        self.backorder_costs = []
        self.backorder = 0
        self.costs = []
        self.arrivals = [] """
        super().reset()
        self.amount_sold = []
        self.amount_sold_total = 0

    def update_all_demand(self, num_demands: int, t) -> None:
        """Updates inv level and pos for all items for curr and upstream

        Args:
            num_demands (int): amount requested
            t (int): time stamp
        """
        amount_sold = self.update_demand(num_demands)
        self.amount_sold.append(amount_sold)
        self.amount_sold_total += amount_sold
        for item in self.inv_level:
            self.check_s(item, t)
    
    def calculate_profit(self):
        return self.amount_sold_total * self.selling_price - super().get_total_cost()

    def calculate_curr_profit(self, t):
        return self.amount_sold[t] * self.selling_price - super().get_curr_total_costs(t)
    
    def __repr__(self):
        return "Retailer({})".format(self.name)




    

        