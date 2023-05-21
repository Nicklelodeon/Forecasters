def Main():

    def __init__(self, state, demand):
        self.state = state
        self.demand = demand
    

    def simulate(self, demand, t):
        self.state.create_state()
        for i in range(t):
            self.state.update_state(demand, i)
        

