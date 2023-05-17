from Demandable import Demandable
from Item import Item
from State import State
import numpy as np

np.random.seed(1234)

new_demandable = Demandable(10, 100, 50, 100)
curr_state = State(new_demandable)
print("OK")

state = curr_state.create_state([-1,0, 1, 1, 2, 2])




