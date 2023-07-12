from RLDemandable import RLDemandable
from RLSupplier import RLSupplier
from RLRetailer import RLRetailer
from RLDistributionCenter import RLDistributionCenter

class RLBasic(RLDemandable):
    def __init__(self, name):
        super().__init__(name, 0, 0, 0, 0)
    
    def make_supplier(self):
        """Converts basic to supplier

        Returns:
            Supplier
        """
        return RLSupplier(self.name)
    
    def make_retailer(self):
        """Converts basic to retailer

        Returns:
            retailer
        """
        return RLRetailer(self.name) 

    def make_distcentre(self):
        """Converts basic to distribution centre

        Returns:
            Distribution Centre
        """
        return RLDistributionCenter(self.name)
    
    def define_demandable(self):
        """Converts basic demandables to respective stakeholders
        based on the location in the network

        Returns:
            Demandable: Supplier, Retailer or Distribution Centre
        """
        if not self.upstream: ## Upstream is empty return supplier
            return self.make_supplier()
        elif not self.downstream: ## Downstream is empty return retailer
            return self.make_retailer()
        else: ## Has both upstream and downstream return distcentre
            return self.make_distcentre()

    def __str__(self):
        return self.name

