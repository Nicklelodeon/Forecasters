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
# data = pd.read_csv("src/6_24months_mldata.csv")
# data = pd.read_csv("src/72_onlinemldata.csv")
# data = pd.read_csv("src/144_mldata.csv.csv")

target = data['profit']
predictors = data.drop(['profit', data.columns[0]], axis=1)

print(predictors)
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.33, random_state=42)
n_cols = predictors.shape[1]


# target = data.df['profit']
# predictors = data.df.drop(['profit'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.33, random_state=42)

# regr = linear_model.LinearRegression()
# regr.fit(X_train, y_train)

# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)

# preds = regr.predict(X_test)
# print(MSE(y_test, preds))

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

X_test = sm.add_constant(X_test)
preds = model.predict(X_test.astype(float))

df = pd.read_csv("./src/TOTALSA.csv")
mean = df['TOTALSA'].mean()
std = df['TOTALSA'].std()

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

def vif_analysis(data):
    """VIF analysis on the data dataframe"""
    for i in range(len(data.columns)):
        v=vif(np.matrix(data),i)
        print("Variance inflation factor for {}: {}".format(data.columns[i],round(v,2)))

vif_analysis(X_train)





print(model.summary())
print("mse: " + str(MSE(y_test, preds)))
print("mae: " + str(MAE(y_test, preds)))
print("mape: " + str(MAPE(y_test, preds)))

