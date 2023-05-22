from Demandable import Demandable
from Supplier import Supplier
from Retailer import Retailer
from DistributionCenter import DistributionCenter


class Basic(Demandable):
    def __init__(self, name):
        super().__init__(name, 0, 0, 0 ,0 ,0)
    
    def make_supplier(self):
        return Supplier(self.name)
    
    def make_retailer(self):
        return Retailer(self.name, 100) 
    
    def make_distcentre(self):
        return DistributionCenter(self.name)
    
    def define_demandable(self):
        if not self.upstream: ## Upstream is empty return supplier
            return self.make_supplier()
        elif not self.downstream: ## Downstream is empty return retailer
            return self.make_retailer()
        else: ## Has both upstream and downstream return distcentre
            return self.make_distcentre()

    def __str__(self):
        return self.name

