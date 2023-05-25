import numpy as np
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly

class GeneticAlgoState(State):
    def __init__(self):
        super().__init__()
        self.demand_class = GenerateDemandMonthly()
        self.iteration = 30
        self.demand_class.simulate_normal(self.iteration)
        self.demand_temp = np.array(self.demand_class.get_demand())
        self.demand_matrix = np.reshape(self.demand_temp, (self.iteration, 12))
    
    def GArun(self, X):
        if not self.check_valid(X):
            return -np.inf
        for j in range(len(self.changeable_network)):
            small_s = X[2 * j]
            big_S = X[2 * j + 1]
            demandable = self.changeable_network[j]
            demandable.change_order_point(small_s, big_S)
        #self.demand_class.simulate_normal(iterations)
        #self.demand_list = self.demand_class.get_demand()
        
        totalsum = 0
        for k in range(len(self.demand_matrix)):
            self.demand_list = self.demand_matrix[k]
            self.reset()
            #print(self.demand_list)
            
            
            for i in range(len(self.demand_list)):
                self.update_state(i)
            
            totalsum += self.total_sum()

        return totalsum / self.iteration
    
    def total_sum(self):
        return sum(self.rewards)
    
    def check_valid(self, X):
        for i in range(len(X)//2):
            index = i * 2
            if X[index] >= X[index + 1]:
                return False
        return True