import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from MLGenerateData import MLGenerateData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import numpy as np 
from State import State
import random

data = pd.read_csv("src/6_24months_US_car_data.csv")

# use profit as dependent variables
target = data['profit']
# use the rest of the variables as independent variables
predictors = data.drop(['profit', data.columns[0]], axis=1)

# 7:3 train-test split 
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
n_cols = predictors.shape[1]

# create OLS model
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

X_test = sm.add_constant(X_test)
preds = model.predict(X_test.astype(float))

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

def test(model, mean, std):
    """simulate 1000000 random (s, S) policies and find the (s, S) with highest predicted profit

    Args:
        model (LinearRegressionModel): linear regression model to be trained
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
        preds = model.predict(params)
        if preds > curr_max:
            curr_max = preds
            lst = [s_DC1, S_DC1, s_DC2, S_DC2, s_r1, S_r1]

    return [curr_max, lst]


print(test(model, mean, std, tests))

# find vif of each independent variable
def vif_analysis(data):
    """returns vif of each indpendent variable
    """
    for i in range(len(data.columns)):
        v=vif(np.matrix(data),i)
        print("Variance inflation factor for {}: {}".format(data.columns[i],round(v,2)))

vif_analysis(X_train)

print(model.summary())
print("mse: " + str(MSE(y_test, preds)))
print("mae: " + str(MAE(y_test, preds)))
print("mape: " + str(MAPE(y_test, preds)))

