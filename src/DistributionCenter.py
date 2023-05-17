from Demandable import Demandable

class DistributionCenter(Demandable):
    def __init__(self, name, selling_price):
        super().__init__(name, 3, 10, 30, 4500, 7500)

        