from Demandable import Demandable
from Item import Item
import numpy as np

np.random.seed(1234)

a = Demandable(10, 100, 50, 100)
b = Demandable(20, 100, 50, 100)
c = Demandable(20, 100, 50, 100)

item1 = Item(str(np.random.randint(1, 1000)), 10)
item2 = Item(str(np.random.randint(1, 1000)), 10)
print(item1)
print(item2)
a.add_item(item1, np.random.randint(4000, 7000))
b.add_item(item1, np.random.randint(4000, 7000))
a.add_item(item2, np.random.randint(4000, 7000))
c.add_item(item2, np.random.randint(4000, 7000))

a.add_upstream(b)
a.add_upstream(c)

# b, c --> a

print(a.get_totalhc())

for i in range(10):
    a.demand(1000, i)
    print("cycle ", i, "a inventory level", a.inv_level)
    print("cycle ", i, "b inventory level", b.inv_level)
    print("cycle ", i, "c inventory level", c.inv_level)
