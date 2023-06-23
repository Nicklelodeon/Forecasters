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
        """ self.function = lambda x: np.floor(
            np.random.triangular(self.low, self.mode, self.high, x)) """
        
    def get_lead_time(self):
        return np.floor(np.random.triangular(self.low, self.mode, self.high, 1)[0])
        #return self.function(1)[0]
    
    def get_expected_value(self):
        list = np.floor(np.random.triangular(self.low, self.mode, self.high, 1000000))
        #list = self.function(1000000)
        return np.round(np.mean(list),2)
    
    def visual(self):
        #lst = np.array(self.function(1000))
        lst = np.array(np.floor(np.random.triangular(self.low, self.mode, self.high, 1000)))
        df = pd.DataFrame(lst, columns = ["values"])
        df = df["values"].value_counts()/1000
        new_df = pd.DataFrame(df)
        sns.barplot(x=new_df.index, y=new_df["count"])
        plt.title("Probability mass function")
        plt.xlabel("Lead time")
        plt.ylabel("Probability")
        plt.show()