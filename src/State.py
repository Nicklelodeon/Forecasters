def State():

    def __init__(self, root):
        self.root = root

    def create_state(self, demandables):
        a = Demandable(10, 100, 50, 100)
        b = Demandable(20, 100, 50, 100)
        c = Demandable(20, 100, 50, 100)
        d = Demandable(30, 100, 50, 100)

        a.add_upstream(b)
        a.add_upstream(c)
        b.add_upstream(d)
        list_end_upstream = a.find_end_upstream()

        for end_demandable in list_end_upstream:
            rand_item = Item(str(np.random.randint(1, 1000)), 10)
            end_demandable.add_item_downstream(rand_item)


    def update_state(self, demand, t):
        self.root.update_all_states(t)
        self.root.update_all_demand(demand)
        self.root.update_all_order(t)
        self.root.update_all_cost(t)

    
    



