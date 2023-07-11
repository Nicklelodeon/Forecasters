import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Stochastic_Lead_Time:
    
    def __init__(self):
        self.distribution = "Triangle"
        self.low = 1
        self.high = 5
        self.mode = 2.5
        
    def get_lead_time(self):
        """Samples integer from distribution

        Returns:
            int: lead time
        """
        return np.floor(np.random.triangular(self.low, self.mode, self.high, 1)[0])
    
    def get_expected_value(self):
        """Expected value of the distribution solved analytically

        Returns:
            float: expected value
        """
        return 7/3
    
    def visual(self):
        """Creates the pmf
        """
        samples = 1000000
        lst = np.array(np.floor(np.random.triangular(self.low, self.mode, self.high, samples)))
        df = pd.DataFrame(lst, columns = ["values"])
        df = df["values"].value_counts()/samples
        new_df = pd.DataFrame(df)
        sns.barplot(x=new_df.index, y=new_df["count"])
        plt.title("Probability mass function")
        plt.xlabel("Lead time")
        plt.ylabel("Probability")
        plt.show()