from Demandable import Demandable
from Item import Item

a = Demandable(100, 100, 50, 100)
b = Demandable(200, 100, 50, 100)
a.add_upstream(b)
print(a.get_totalhc())


item_a = Item("a", 10)
item_a2 = Item("a", 10)
print(item_a == item_a2)