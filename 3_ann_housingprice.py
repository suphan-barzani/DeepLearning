
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()

# Tenemos que normalizar los datos

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

x_train = train_data
y_train = train_targets

x_test = test_data
y_test = test_targets

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])

results = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=1,
                    validation_data=(x_test, y_test))

history = results.history

loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(loss)+1)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

mae = history['mae']
val_mae = history['val_mae']

plt.figure()

plt.plot(epochs, mae, 'bo', label='Training MAE')
plt.plot(epochs, val_mae, 'b', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

y_pred = model.predict(x_test)[:, 0]

import numpy as np

r2 = np.corrcoef(y_test, y_pred)[1, 0]**2

plt.figure()
plt.scatter(y_test, y_pred)
plt.title(f'R$^2$ = {r2}')

plt.show()
