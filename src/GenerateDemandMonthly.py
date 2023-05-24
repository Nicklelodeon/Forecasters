import math
import numpy as np
import scipy.stats


class GenerateDemandMonthly:
    def __init__(self):
        self.demand = []
        self.quantiles = []

    def simulate_normal(self, year, mean=20):
        """Creates random demand with a monthly demand with z amt of years
        
        Args:
            year (int): Number of years
        """
        return_demand = []
        for z in range(year):
            for i in range(1, 13):
                if i in [1, 2, 3, 5, 6, 10]:
                    demand, quantile = self.normal_distribution(mean * 0.75, math.sqrt(mean * 0.75), 1)
                    self.demand.extend(demand)
                    return_demand.extend(demand)
                    self.quantiles.extend(quantile)
                else:
                    demand, quantile = self.normal_distribution(mean * 1.5, math.sqrt(mean * 1.5), 1)
                    self.demand.extend(demand)
                    return_demand.extend(demand)
                    self.quantiles.extend(quantile)
        
        return return_demand

    
    def normal_distribution(self, mean, sd, size):
        arr = [round(x) for x in np.random.normal(mean, sd, size)]
        dist = scipy.stats.norm.ppf( [.25, .75], mean, sd)
        return (arr, dist)

    def simulate_poisson(self, year, mean=20):
        """Creates random demand with a monthly demand with z amt of years
        
        Args:
            year (int): Number of years
        """
        return_demand = []
        for z in range(year):
            for i in range(1, 13):
                if i in [1, 2, 3, 5, 6, 10]:
                    demand, quantile = self.poisson_distribution(0.75 * mean, 1)
                    self.demand.extend(demand)
                    return_demand.extend(demand)
                    self.quantiles.extend(quantile)
                else:
                    demand, quantile = self.poisson_distribution(1.5 * mean, 1)
                    self.demand.extend(demand)
                    return_demand.extend(demand)
                    self.quantiles.extend(quantile)

        return return_demand
    
    def poisson_distribution(self, mean, size):
        arr = [round(x) for x in np.random.poisson(mean, size)]
        dist = scipy.stats.poisson.ppf( [.25, .75], mean)
        return (arr, dist)

    


    def get_demand(self):
        return self.demand

