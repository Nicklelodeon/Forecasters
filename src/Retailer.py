from Demandable import Demandable

class Retailer(Demandable):
    def __init__(self, holding_cost):
        super().__init__(holding_cost)
        self.backorder = 0
    
print(1)
        