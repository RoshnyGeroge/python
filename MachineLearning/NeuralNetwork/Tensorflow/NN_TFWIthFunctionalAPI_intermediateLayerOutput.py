# reference https://www.youtube.com/watch?v=pAhPiF3yiXI&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=3
# handwritten digits 0-9  recognition from images of 28x28 size
# using sequential model with Functional API
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Use the below two lines incase need to use GPU
# physical_devices = tf.config.list_physical_devices
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("X_train shape ", x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(-1, 28*28).astype("float32")/255
# matrix.reshape(-1,x) reduces the dimension of the array by specified order.
# here 3D array reshaped to 2D array
# 28 x 28 array is converted to 784 single dimension array.
# int32 array is converted to float 32 for ease of computation.
#  it is a grey scale image, hence division by 255 for normalization.

x_test = x_test.reshape(-1,28*28).astype("float32")/255

print("X_train after Reshape ", x_train.shape)
print(y_test.shape)
# these are numpy arrays now.  the tf will conversion to tensor internally.

# Functional API( A bit more flexible)


# other way to specify model that will help to debug the hiddden layers.
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation= 'relu'))
model.add(layers.Dense(256, activation = 'relu', name='mylayer'))
model.add(layers.Dense(10))
# get insights on one hidden layer
#model = keras.Model(inputs= model.inputs, outputs = model.layers[-2].output)
#feature = model.predict(x_test)
#print(feature.shape)

# get insight on all hidden layers
model = keras.Model(inputs= model.inputs, outputs = [layer.output for layer in model.layers])

features = model.predict(x_test)

for feature in features:
    print(feature.shape)



import sys
sys.exit()


inputs = keras.Input(shape=(784))
x=layers.Dense('512', activation = 'relu')(inputs)
x=layers.Dense('256', activation = 'relu')(x)
outputs=layers.Dense('10', activation= 'softmax')(x)

model= keras.Model(inputs=inputs, outputs=outputs)

model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam( learning_rate = 0.001),
        metrics=["accuracy"],
    )

print ("Training the model:")
model.fit(x_train, y_train, batch_size =32, epochs = 5, verbose = 2)

print("Evaluate the model:")
model.evaluate(x_test, y_test, batch_size = 32, verbose= 2)

print ("Y-Test[1]", y_test[1])
#model.evaluate(x_test[1], y_test[1], batch_size= 1, verbose= 2)


print (model.summary())