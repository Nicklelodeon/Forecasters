from MLGenerateData import MLGenerateData
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = MLGenerateData()
data.create_data()


target = data.df['profit']
predictors = data.df.drop(['profit'], axis=1)
n_cols = predictors.shape[1]



model = keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(n_cols,)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(layers.Dropout(0.25)))
model.add(layers.Dense(1, activation='relu'))

model.compile(optimiser='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(predictors, target)



