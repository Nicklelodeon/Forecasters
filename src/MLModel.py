from MLGenerateData import MLGenerateData

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

data = MLGenerateData()
data.create_data()


target = data.df['profit']
predictors = data.df.drop(['profit'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.33, random_state=42)
n_cols = predictors.shape[1]



model = keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(n_cols,)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(12, activation='relu'))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='root_mean_squared_error', metrics=['mean_absolute_percentage_error'])
model.fit(X_train, y_train)
#results = model.predict(X_test)
#print(results)
print(model.evaluate(X_test, y_test))



