class Item:
    def __init__(self, name, cost):
        self.name = name.lower()
        self.cost = cost

    def get_name(self):
        return self.name

    def __repr__(self):
        return "Item()"

    def __str__(self):
        return "Item {} has cost of {}".format(self.name, self.cost)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Item):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)
