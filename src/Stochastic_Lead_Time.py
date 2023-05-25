import numpy as np

class Stochastic_Lead_time:
    
    def __init__(self):
        self.distribution = "Triangle"
        self.low = 0.5
        self.high = 4.5
        self.mode = 2
        
    def get_lead_time(self):
        return round(np.random.triangular(self.low, self.mode, self.high, 1)[0])
        
    