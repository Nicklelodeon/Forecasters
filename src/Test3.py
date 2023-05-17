from Demandable import Demandable
from Item import Item
import numpy as np

np.random.seed(1234)

a = Demandable("a", 10, 100, 1000, 4500, 7500)
b = Demandable("b", 20, 100, 1000, 4500, 7500)
c = Demandable("c", 20, 100, 1000, 4500, 7500)
d = Demandable("d", 30, 100, 1000, 4500, 7500)

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
    #print("b cycle ", i, "a inventory level", a.inv_level)
    #print("b cycle ", i, "b inventory level", b.inv_level)
    #print("b cycle ", i, "c inventory level", c.inv_level)
    #print("b cycle ", i, "d inventory level", d.inv_level)
    a.update_all_inventory(i)
    a.update_all_demand(1000, i)
    
    a.update_all_cost(i)
    """
    print("a cycle ", i, "a inventory level", a.inv_level)
    print("a cycle ", i, "b inventory level", b.inv_level)
    print("a cycle ", i, "c inventory level", c.inv_level)
    print("a cycle ", i, "d inventory level", d.inv_level)
    print("order ", i, "a inventory level", a.arrivals)
    print("order ", i, "b inventory level", b.arrivals)
    print("order ", i, "c inventory level", c.arrivals)
    print("d ", i, "d inventory level", d.arrivals)
    print("cost", i, a.get_total_cost(i))
    """
    print(a)
