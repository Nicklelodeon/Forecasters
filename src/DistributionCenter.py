from Demandable import Demandable

class DistributionCenter(Demandable):
    def __init__(self, name):
        super().__init__(name, 3, 5, 10, 50, 100)


    def __repr__(self):
        return "DistributionCenter({})".format(self.name)
