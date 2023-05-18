from Demandable import Demandable

class DistributionCenter(Demandable):
    def __init__(self, name):
        super().__init__(name, 3, 10, 30, 4500, 7500)

    def __repr__(self):
        return "DistributionCenter({})".format(self.name)