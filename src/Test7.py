import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from State import State
from GenerateDemandMonthly import GenerateDemandMonthly

class GeneticAlgoState(State):
    def __init__(self):
        super().__init__()
        self.demand_class = GenerateDemandMonthly()
        self.demand_class.simulate_normal(10)
        self.demand_temp = np.array(self.demand_class.get_demand())
        self.demand_matrix = np.reshape(self.demand_temp, (10, 12))
    
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

        return totalsum / 10
    
    def total_sum(self):
        return sum(self.rewards)
    
    def check_valid(self, X):
        for i in range(len(X)//2):
            index = i * 2
            if X[index] >= X[index + 1]:
                return False
        return True

genstate = GeneticAlgoState()
genstate.create_state([-1,0, 1, 1, 2, 2])


#print(-1 * genstate.GArun([58,69,41,52,49,52]))

def f(X):
    return -1 * genstate.GArun(X)

varbound=np.array([[40,90]]*6)

model=ga(function=f,dimension=6,variable_type='int',variable_boundaries=varbound)


algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model.run()
