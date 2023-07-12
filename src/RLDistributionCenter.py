from RLDemandable import RLDemandable

class RLDistributionCenter(RLDemandable):
    def __init__(self, name):
        super().__init__(name, 1, 5, 40, 90)


    def __repr__(self):
        return "DistributionCenter({})".format(self.name)
