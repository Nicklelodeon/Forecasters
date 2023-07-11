import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class GenerateDemandMonthly:
    def __init__(self):
        self.demand = []
        
    
    def visualize(self):
        """ create pmf for demands
        """
        lst = self.demand
        df = pd.DataFrame(lst, columns = ["values"])
        length = len(df)
        df = df["values"].value_counts()/(length)
        new_df = pd.DataFrame(df)
        sns.barplot(x=new_df.index, y=new_df["count"])
        plt.title("Probability mass function of demands")
        plt.xlabel("Demand value")
        plt.xticks(rotation=45)
        plt.ylabel("Probability")
        plt.show()
        
    def simulate_normal_no_season(self, periods=108, mean=30, std = 4):
        """Creates random normal demand with no season

        Args:
            periods (int): number of periods
            mean (int, optional): mean of normal. Defaults to 30.
            std (int, optional): std of normal. Defaults to 4.
        """
        return_demand = []
        demand = np.round(np.random.normal(mean, std, periods))
        self.demand.extend(demand)
        return_demand.extend(demand)
        return return_demand
        
    def simulate_poisson_no_season(self, periods=108, mean=30):
        """Creates random Poisson demand with no season

        Args:
            periods (int): number of periods
            mean (int, optional): mean of normal. Defaults to 30.
        """
        return_demand = []
        demand = np.round(np.random.poisson(mean, periods))
        self.demand.extend(demand)
        return_demand.extend(demand)
        return return_demand


    def clear(self):
        """resets demand
        """
        self.demand = []


    def get_demand(self):
        """returns list of current demand

        Returns:
            list: curr demand
        """
        return self.demand

