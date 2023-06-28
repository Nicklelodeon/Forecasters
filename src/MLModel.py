
import pandas as pd
import numpy as np 
from State import State
import random

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load("src/DNNmodel.pt"))
model.eval()

tests = [[54, 63, 42, 47, 42, 49], np.round([42.65454086832623, 55.47066141610458, 45.23110339190882 , 69.56550258424139, 30.0, 50.14861820830117])]

def test(model, mean, std, tests):
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
    curr_max = 0
    lst = []
    for x in range(1000000):
        # s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1 = [random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8)), random.randint(round(mean * 2), round(mean * 4)), random.randint(round(mean * 5), round(mean * 8))]
        s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1 = [random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10))]
        if s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1:
            continue

        params = [1, mean, std, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1]
        preds = model.predict(params)
        if preds > curr_max:
            curr_max = preds
            lst = [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1]
    for x in tests:
        params = [1, mean, std]
        params.extend(x)
        preds = model.predict(params)
        if preds > curr_max:
            curr_max = preds
            lst = x

    return [curr_max, lst]

# print('result', model.predict([1, mean, std, 0.0, 54, 63, 42, 47, 42, 49]))

print(test(model, mean, std, tests))
