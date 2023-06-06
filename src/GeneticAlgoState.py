import numpy as np
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly

class GeneticAlgoState(State):
    def __init__(self, iteration = 30):
        super().__init__()
        self.demand_class = GenerateDemandMonthly()
        self.iteration = 100
        #self.demand_class.simulate_normal(self.iteration)
        self.demand_temp = []#np.array(self.demand_class.get_demand())
        self.demand_matrix = []#np.reshape(self.demand_temp, (self.iteration, 12))
        
    def GArun_no_season(self, X):
        demand = self.demand_class.simulate_normal_no_season(periods = 24 * self.iteration)
        
        self.demand_temp = demand
        self.demand_matrix = np.reshape(self.demand_temp, (24, self.iteration))
        
        for j in range(1,len(self.changeable_network)):
            small_s = X[j]
            big_S = X[j + len(self.changeable_network)]
            demandable = self.changeable_network[j]
            demandable.change_order_point(small_s, big_S)
        
        totalsum = 0
        for k in range(len(self.demand_matrix)):
            self.reset(X[0])
            self.set_demand_list(self.demand_matrix[k])

            for i in range(len(self.demand_list)):
                self.update_state(i)
            
            totalsum += self.total_sum()

        return totalsum / self.iteration
        
        
        
    def GArun2(self, X):
        self.make_valid(X)
        for i in range(len(self.demand_matrix)):
            self.reset()
            self.set_demand_list(self.demand_matrix[i])
            
            totalsum = 0
            
            for t in range(len(self.demand_list)):
                for k in range(len(self.changeable_network)):
                    index = k * len(self.demand_list) + 2 * t
                    small_s = X[index]
                    big_S = X[index + 1]
                    demandable = self.changeable_network[k]
                    demandable.change_order_point(small_s, big_S)
                self.update_state(t)
            totalsum += self.total_sum()
        return totalsum / self.iteration

                    
                    
    def GArun(self, X):
        if not self.check_valid(X):
            return -2147483648
        for j in range(len(self.changeable_network)):
            small_s = X[2 * j]
            big_S = X[2 * j + 1]
            demandable = self.changeable_network[j]
            demandable.change_order_point(small_s, big_S)
        
        totalsum = 0
        for k in range(len(self.demand_matrix)):
            self.reset()
            self.set_demand_list(self.demand_matrix[k])

            for i in range(len(self.demand_list)):
                self.update_state(i)
            
            totalsum += self.total_sum()

        return totalsum / self.iteration
    
    def total_sum(self):
        return self.rewards
    
    def check_valid(self, X):
        for i in range(len(X)//2):
            index = i * 2
            if X[index] > X[index + 1]:
                return False
        return True
    
    def make_valid(self, X):
        for i in range(len(X)//2):
            index = i * 2
            if X[index] > X[index + 1]:
                X[index], X[index + 1] = X[index + 1], X[index]