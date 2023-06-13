from Demandable import Demandable

class Retailer(Demandable):
    def __init__(self, name, selling_price):
        super().__init__(name, 2, 6, 40, 90)
        #super().__init__(name, 1, 2, 3, 40, 90)
        self.amount_sold = []
        self.selling_price = selling_price
        self.amount_sold_total = 0
        
    def set_optimal_selling_price(self, multiplier):
        """Sets optimal price * multipler

        Args:
            multiplier (float): multiplier
        """
        self.selling_price = self.find_optimal_cost() * multiplier

    def reset(self, amount = 65):
        super().reset()
        self.amount_sold = []
        self.amount_sold_total = 0
        
    def update_demand(self, num_demands: int):
        """Updates inv level and pos for current

        Args:
            num_demands (int): amount requested
        """
        amount_sold = super().update_demand(num_demands)
        self.amount_sold.append(amount_sold)
        self.amount_sold_total += amount_sold

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
        #print("self amount sold t:", self.amount_sold[t], "super get curr total costs:", super().get_curr_total_costs(t))
        return self.amount_sold[t] * self.selling_price - self.get_curr_cost(t)
    
    def __str__(self):
        return super().__str__() + "\n" + "unit sold: " + \
        str(self.amount_sold[max(0, len(self.amount_sold) - 1)]) + \
        " price sold: " + str(self.amount_sold[max(0, len(self.amount_sold) - 1)] *self.selling_price)
    
    def __repr__(self):
        return "Retailer({})".format(self.name)




    

        