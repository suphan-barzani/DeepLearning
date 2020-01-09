
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data()

# Comprobar tamaño y dimensiones en iPython. Ploteamos los datos para ver cómo
# son y en qué intervalos 0-255 se mueve

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

# Para que una red neuronal en Keras funcione, necesitamos seleccionar tres
# cosas más, como parte del proceso de compilación:
#
# Una función de pérdida - Cómo la red es capaz de medir su rendimiento sobre
# los datos de entrenamiento, y ver hacia dónde tiene que dirigirse
#
# Un optimizador - Mecanismo a través del cuál la red se actualizará a sí misma
# en base a los datos observados y a la función de pérdida
#
# La métrica que se va a monitorizar - El valor estadístico a utilizar como
# resultado del proceso de entrenamiento y testeo

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preprocesamos los datos

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# Preparamos las muestras

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluamos ahora el conjunto de test

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'test_acc: {test_acc}')
