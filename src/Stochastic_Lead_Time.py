import numpy as np

class Stochastic_Lead_time:
    
    def __init__(self):
        self.distribution = "Triangle"
        self.low = 0.5
        self.high = 4.5
        self.mode = 2
        
    def get_lead_time(self):
        print("jere")
        return round(np.random.triangular(0, 1, 5, 1)[0])

stl = Stochastic_Lead_time()
print(stl.get_lead_time())

round(np.random.triangular(0, 1, 5, 1000))
        
    