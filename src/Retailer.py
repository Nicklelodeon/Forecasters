from Demandable import Demandable

class Retailer(Demandable):
    def __init__(self, name, selling_price):
        super().__init__(name, 5, 15, 50, 4500, 7500)
        self.amount_sold = 0
        self.selling_price = selling_price

    def update_all_demand(self, num_demands: int, t) -> None:
        """Updates inv level and pos for all items for curr and upstream

        Args:
            num_demands (int): amount requested
            t (int): time stamp
        """
        amount_sold = self.update_demand(num_demands)
        self.amount_sold += amount_sold
        for item in self.inv_level:
            self.check_s(item, t)
    
    def calculate_profit(self, t):
        return self.amount_sold * self.selling_price - super.arrivals[t]
    
    def __repr__(self):
        return "Retailer({})".format(self.name)




    

        