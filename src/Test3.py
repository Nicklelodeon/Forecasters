from Demandable import Demandable
from Item import Item
import numpy as np

np.random.seed(1234)

a = Demandable(10, 100, 50, 100)
b = Demandable(20, 100, 50, 100)
c = Demandable(20, 100, 50, 100)
d = Demandable(30, 100, 50, 100)

a.add_upstream(b)
a.add_upstream(c)
b.add_upstream(d)

print(a.upstream)
#print(b.downstream[0] == a)
list_end_upstream = a.find_end_upstream()

for end_demandable in list_end_upstream:
    rand_item = Item(str(np.random.randint(1, 1000)), 10)
    end_demandable.add_item_downstream(rand_item)

for i in range(10):
    a.demand(1000, i)
    print("cycle ", i, "a inventory level", a.inv_level)
    print("cycle ", i, "b inventory level", b.inv_level)
    print("cycle ", i, "c inventory level", c.inv_level)
    print("cycle ", i, "d inventory level", d.inv_level)
