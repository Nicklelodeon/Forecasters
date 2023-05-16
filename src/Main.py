def Main():

    def __init__(self, demandables):
        self.demandables = {}
        for i in range(len(demandables)):
            self.demandables[i] = demandables[i]
    

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


    def update_state(self, t):


