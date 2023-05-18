from Demandable import Demandable
from Item import Item
import numpy as np
from State import State
from SyntheticDemand import GenerateDemand

synthetic = GenerateDemand()
synthetic.simulate(3) #Simulating 3 years
demand_list = synthetic.get_demand()

print(demand_list)
