from State import State 

class BayesianState(State):
    def __init__(self):
        super().__init__()

    def update_state(self, t):
        """Discrete update state

        Args:
            demand (_type_): _description_
            t (int): time
        """
        
        self.root.update_all_inventory(t)
        self.root.update_all_demand(self.demand_list[t], t)
        self.root.update_all_cost(t)
        self.rewards.append(self.root.calculate_profit(t))