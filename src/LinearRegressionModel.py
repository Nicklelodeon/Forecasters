import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from MLGenerateData import MLGenerateData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


data = MLGenerateData()
data.create_data()

largest = max(data.df['profit'])
if abs(min(data.df['profit'])) > largest:
    largest = abs(min(data.df['profit']))
target = data.df['profit'] / largest
predictors = data.df.drop(['profit'], axis=1)
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

print(model.summary())

