
import pandas as pd
import numpy as np 
from State import State
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchmetrics import MeanAbsolutePercentageError

data = pd.read_csv("./src/6_24months_US_car_data.csv")

# data cleaning and transformation
target_df = data['profit']
target = torch.tensor(target_df)
predictors = torch.tensor(data.drop(['profit'], axis=1).to_numpy(dtype=np.float64))
X_train_raw, X_test_raw, y_train, y_test = train_test_split(predictors, target, test_size=0.33, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
n_cols = predictors.shape[1]

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

# build a model with 7 Linear layers and ReLU activation function
model = nn.Sequential(
    nn.Linear(n_cols, 800),
    nn.ReLU(),
    nn.Linear(800, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 1)
)

# load saved ML model
model.load_state_dict(torch.load("src/model_mse.pt"))
model.eval()


def test(model, mean, std):
    """simulate 1000000 random (s, S) policies and find the (s, S) with highest predicted profit

    Args:
        model (Sequential): Neural Network model to be trained
        mean (float): mean of the demand distribution
        std (float): std of the demand distribution

    Returns:
        list: returns list of highest profit and nested list containing the corresponding (s, S) policy
    """
    state = State()
    state.create_state([-1 ,0, 1, 1, 2, 2], mean=mean, std=std)
    curr_max = 0
    lst = []
    for x in range(1000000):
        s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1 = [random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10)), random.randint(round(mean * 2), round(mean * 10))]
        if s_DC1 >= S_DC1 or s_DC2 >= S_DC2 or s_r1 >= S_r1:
            continue

        params = [1, mean, std, s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1]
        preds = model(torch.tensor(np.array(params, dtype=np.float64)).float())
        if preds > curr_max:
            curr_max = preds
            lst = [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1]

    return [curr_max, lst]

print(test(model, mean, std))
