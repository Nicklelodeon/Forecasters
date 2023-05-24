from State import State 

class MLState(State):
    def __init__(self):
        super().__init__()
    
    def change_demand(self, demand_list):
        self.demand_list = demand_list