import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)


class Stochastic_Lead_Time:
    
    def __init__(self):
        self.distribution = "Triangle"
        self.low = 1
        self.high = 5
        self.mode = 2.5
        self.function = lambda x: np.floor(
            np.random.triangular(self.low, self.mode, self.high, x))
        
    def get_lead_time(self):
        return self.function(1)[0]
    
    def get_expected_value(self):
        list = self.function(1000000)
        return np.round(np.mean(list),2)
    
    def visual(self):
        list = self.function(1000)
        plt.hist(list, bins = 50, density = True)
        plt.show()