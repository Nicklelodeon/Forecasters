from Demandable import Demandable
from Item import Item

a = Demandable(10, 100, 50, 100)
b = Demandable(20, 100, 50, 100)
c = Demandable(20, 100, 50, 100)

a.add_upstream(b)
a.add_upstream(c)

# b, c --> a 

print(a.get_totalhc())

for i in range(10):
    a.demand(1000, i)
    print("cycle ", i, "a inventory level",a.inv_level)
    print("cycle ", i, "b inventory level",b.inv_level)
    print("cycle ", i, "c inventory level",c.inv_level)

