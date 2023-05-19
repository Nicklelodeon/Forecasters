import math
import numpy as np


class GenerateDemandMonthly:
    def __init__(self):
        self.demand = []

    def simulate(self, year):
        """Creates random demand with a monthly demand with z amt of years
        
        Args:
            year (int): Number of years
        """
        for z in range(year):
            for i in range(1, 13):
                
                if i in [1, 2, 3, 5, 6, 10]:
                    self.demand.extend(
                        [round(x) for x in np.random.normal(50, math.sqrt(25), 1)]
                    )
                
                else:
                    self.demand.extend(
                        [round(x) for x in np.random.normal(65, math.sqrt(49), 1)]
                    )
    
    def get_demand(self):
        return self.demand
abc = GenerateDemandMonthly()
abc.simulate(1)
print(abc.get_demand())