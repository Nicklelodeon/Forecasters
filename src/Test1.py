from Demandable import Demandable
from Item import Item
import numpy as np

np.random.seed(1234)

print(Item("123", 50))

a = Demandable(10, 100, 50, 100)
b = Demandable(20, 100, 50, 100)
item1 = Item(str(np.random.randint(1, 1000)), 10)

a.add_item(item1, np.random.randint(4000, 7000))
b.add_item(item1, np.random.randint(4000, 7000))
a.add_upstream(b)
print(a.get_totalhc())

# b --> a

for i in range(10):
    a.demand(1000, i)
    print("cycle ", i, "a inventory level", a.inv_level)
    print("cycle ", i, "b inventory level", b.inv_level)
