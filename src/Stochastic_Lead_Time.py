import numpy as np

class Stochastic_Lead_Time:
    
    def __init__(self):
        self.distribution = "Triangle"
        self.low = 1
        self.high = 5
        self.mode = 2.5
        
    def get_lead_time(self):
        return np.floor(np.random.triangular(self.low, self.mode, self.high, 1)[0])
