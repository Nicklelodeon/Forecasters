import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from MLGenerateData import MLGenerateData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import numpy as np 

data = pd.read_csv("src/6_onlinemldata.csv")
# data = pd.read_csv("src/6_24months_mldata.csv")
# data = pd.read_csv("src/72_onlinemldata.csv")
# data = pd.read_csv("src/144_mldata.csv.csv")

target = data['profit']
predictors = data.drop(['profit'], axis=1)
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

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

print(model.summary())
print("mse: " + str(MSE(y_test, preds)))
print("mae: " + str(MAE(y_test, preds)))
print("mape: " + str(MAPE(y_test, preds)))
print("smape: " + str(smape(y_test, preds)))

