from Demandable import Demandable

class Basic(Demandable):
    def __init__(self, name):
        super().__init__(name, 0, 0, 0 ,0 ,0 ,0)
    
    def make_supplier(self):
        return Supplier(self.name)
    
    def make_retailer(self):
        return Retailer(self.name)
    
    def make_distcentre(self):
        return Distribution_Centre(self.name)
    
    def define_demandable(self):
        if not self.upstream: ## Upstream is empty return supplier
            return self.make_supplier()
        elif not self.downstream: ## Downstream is empty return retailer
            return self.make_retailer()
        else: ## Has both upstream and downstream return distcentre
            return self.make_distcentre()

