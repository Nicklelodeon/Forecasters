import pandas as pd
import numpy as np

rl_data = pd.read_csv('src\RL_Dataframe.csv', index_col=0)
rl = rl_data['RL_no_season'].tolist()
rl_24 = rl_data['RL_no_season_24'].tolist()
rl_poisson = rl_data['RL_poisson_no_season'].tolist()
rl_poisson_lead_time = rl_data['RL_no_season_poisson_lead_time'].tolist()
rl_both_poisson = rl_data['RL_poisson_no_season_poisson_lead_time'].tolist()
