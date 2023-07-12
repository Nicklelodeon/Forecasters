class Item:
    def __init__(self, name, cost):
        self.name = name.lower()
        self.cost = cost

    def __repr__(self):
        return "Item({})".format(self.name)

    def __str__(self):
        return "Item {} has cost of {}".format(self.name, self.cost)

    def __eq__(self, other):
        """Overrides original equals method, 2 items are equal if they have same name and cost"""
        if isinstance(other, Item):
            return self.name == other.name and self.cost == other.cost
        return False

    def __hash__(self):
        return hash(self.name)

    def get_name(self):
        """returns name of item

        Returns:
            string: name of item
        """
        return self.name

    def get_cost(self):
        """returns cost of item

        Returns:
            float: cost of item
        """
        return self.cost

    
