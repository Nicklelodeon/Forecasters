import math
import numpy as np
import scipy.stats


class GenerateDemandMonthly:
    def __init__(self):
        self.demand = []
        self.quantiles = []

    def simulate(self, year):
        """Creates random demand with a monthly demand with z amt of years
        
        Args:
            year (int): Number of years
        """
        for z in range(year):
            for i in range(1, 13):
                
                if i in [1, 2, 3, 5, 6, 10]:
                    arr = [round(x) for x in np.random.normal(20, math.sqrt(20), 1)]
                    self.demand.extend(arr)
                    dist = scipy.stats.norm.ppf( [.25, .75], 20, math.sqrt(20)) 
                    self.quantiles.append(dist)
                
                else:
                    arr = [round(x) for x in np.random.normal(30, math.sqrt(30), 1)]
                    self.demand.extend(arr)
                    dist = scipy.stats.norm.ppf( [.25, .75], 30, math.sqrt(30)) 
                    self.quantiles.append(dist)
    
    def get_demand(self):
        return self.demand

