import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Poisson_Stochastic_Lead_Time:
    
    def __init__(self):
        self.distribution = "Shifted Poisson"
        self.function = lambda x: np.random.poisson(4/3, x)

    def get_lead_time(self):
        return self.function(1)[0] + 1
    
    def get_expected_value(self):
        return 7/3
    
    def visual(self):
        lst = np.array(self.function(1000000))
        df = pd.DataFrame(lst, columns = ["values"])
        df = df["values"].value_counts()/1000000
        new_df = pd.DataFrame(df)
        sns.barplot(x=new_df.index, y=new_df["count"])
        plt.title("Probability mass function")
        plt.xlabel("Lead time")
        plt.ylabel("Probability")
        plt.show()
