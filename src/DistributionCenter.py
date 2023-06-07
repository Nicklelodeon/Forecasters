from Demandable import Demandable

class DistributionCenter(Demandable):
    def __init__(self, name):
        super().__init__(name, 1, 2, 40, 90)


    def __repr__(self):
        return "DistributionCenter({})".format(self.name)
