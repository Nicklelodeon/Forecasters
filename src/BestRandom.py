from Demandable import Demandable
from Item import Item
from BayesianState import BayesianState
import numpy as np
from GenerateDemandMonthly import GenerateDemandMonthly 
from State import State
import pandas as pd 

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()


def validate(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1):
    """tests the profit generated by specified (s, S) policy 

    Args:
        s_DC1 (int)
        S_DC1 (int)
        s_DC2 (int)
        S_DC2 (int)
        s_r1 (int)
        S_r1 (int)

    Returns:
        int: mean profit earned over 500 iterations
    """
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
    return (np.mean(state.run(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1)))
    

def random_sample_validate():
    """samples random (s, S) policies and keeps track of the highest profit and corresponding (s, S) policy 

    Returns:
        list: list containing the highest profit and corresponding (s, S) policy 
    """
    max_profit = 0
    returned_ss = []
    bottom = round(mean * 2)
    top = round(mean * 10)
    lst = []
    np.random.seed(9871)
    list_ss = []
    for i in range(500):
        for j in range(3):
            p1 = np.random.randint(bottom, top)
            list_ss.append(p1)
            p2 = np.random.randint(p1, top + 1)
            list_ss.append(p2)
    list_ss = np.array(list_ss)
    matrix_ss = np.reshape(list_ss, (500, 6))
    
    for ss_row in matrix_ss:
        s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1 = ss_row
        curr = validate(s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1)
        if curr > max_profit:
            max_profit = curr
            returned_ss = ss_row
            print(returned_ss)
    return [returned_ss, max_profit]


print(random_sample_validate())
