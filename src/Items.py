class Items:
    def __init__(self):
        self.item_list = set()

    def add(self, item):
        self.item_list.add(item)

    def remove(self, item):
        try:
            self.item_list.remove(item)
        except KeyError:
            print("Invalid item!")

    def contains(self, item):
        return item in self.item_list

    def size(self):
        return len(self.item_list)

    def __str__(self):
        string = ""
        for i in self.item_list:
            string += i.get_name() + " "
        return string
