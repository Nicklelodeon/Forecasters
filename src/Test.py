from Demandable import Demandable
from Item import Item

a = Demandable(100, 100, 50, 100)
b = Demandable(200, 100, 50, 100)
a.add_upstream(b)
print(a.get_totalhc())

# b --> a 

for i in range(10):
    a.demand(1000, i)
    print("cycle ", i, "a inventory level",a.inv_level)
    print("cycle ", i, "b inventory level",b.inv_level)


    

