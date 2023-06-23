import numpy as np
import pandas as pd

from State import State
from GenerateDemandMonthly import GenerateDemandMonthly

np.random.seed(7890)
#### Generate New Demand ####
# demand_generator = GenerateDemandMonthly()

# period = 108
# iterations = 100


# reward = []

state = State()
state.create_state([-1,0,1,1,2,2])

# state.test_no_season(51, 52, 44, 58, 45, 47)
state.run_concurrent(51, 52, 44, 58, 45, 47)