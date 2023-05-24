import math
import numpy as np
import scipy.stats


class GenerateDemandMonthly:
    def __init__(self):
        self.demand = []
        self.quantiles = []

    def simulate_normal(self, year):
        """Creates random demand with a monthly demand with z amt of years
        
        Args:
            year (int): Number of years
        """
        for z in range(year):
            for i in range(1, 13):
                
                if i in [1, 2, 3, 5, 6, 10]:
                    demand, quantile = self.normal_distribution(20, math.sqrt(20), 1)
                    self.demand.extend(demand)
                    self.quantiles.extend(quantile)
                else:
                    demand, quantile = self.normal_distribution(30, math.sqrt(30), 1)
                    self.demand.extend(demand)
                    self.quantiles.extend(quantile)

    
    def normal_distribution(self, mean, sd, size):
        arr = [round(x) for x in np.random.normal(mean, sd, size)]
        dist = scipy.stats.norm.ppf( [.25, .75], mean, sd)
        return (arr, dist)

    def simulate_poisson(self, year):
        """Creates random demand with a monthly demand with z amt of years
        
        Args:
            year (int): Number of years
        """
        for z in range(year):
            for i in range(1, 13):
                
                if i in [1, 2, 3, 5, 6, 10]:
                    demand, quantile = self.poisson_distribution(20,1)
                    self.demand.extend(demand)
                    self.quantiles.extend(quantile)
                else:
                    demand, quantile = self.normal_distribution(30, 1)
                    self.demand.extend(demand)
                    self.quantiles.extend(quantile)
    
    def poisson_distribution(self, mean, size):
        arr = [round(x) for x in np.random.poisson(mean, size)]
        dist = scipy.stats.poisson.ppf( [.25, .75], mean)
        return (arr, dist)

    


    def get_demand(self):
        return self.demand

