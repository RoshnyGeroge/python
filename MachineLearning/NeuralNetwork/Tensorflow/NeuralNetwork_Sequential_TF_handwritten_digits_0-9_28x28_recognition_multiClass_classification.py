# reference https://www.youtube.com/watch?v=pAhPiF3yiXI&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=3
# handwritten digits 0-9  recognition from images of 28x28 size
# using sequential model
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
# matrix.reshape(-1,x) reduces the dimension of the array by 1.
# here 3D array reshaped to 2D array
# 28 x 28 array is converted to 784 single dimension array.
# int32 array is converted to float 32 for ease of computation.
#  it is a grey scale image, hence division by 255 for normalization.

x_test = x_test.reshape(-1,28*28).astype("float32")/255

print("X_train after Reshape ", x_train.shape)
print(y_test.shape)
# these are numpy arrays now.  the tf will conversion to tensor internally.

# sequential API( very convenient , not very flexible)
model = keras.Sequential(
    [
        layers.Dense('512', activation = 'relu'),
        layers.Dense('512', activation = 'relu'),
        layers.Dense('10',)

    ]
)
model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #optimizer=keras.optimizers.Adam( learning_rate = 0.001),
        #optimizer=keras.optimizers.Adagrad(learning_rate=0.001),
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["accuracy"],
    )

print ("Training the model:")
model.fit(x_train, y_train, batch_size =32, epochs = 20, verbose = 2)

print("Evaluate the model:")
model.evaluate(x_test, y_test, batch_size = 32, verbose= 2)

#print ("Y-Test[1]", y_test[1])


# Result with 20 epochs with optimizer Adam
#1875/1875 - 13s - loss: 0.0118 - accuracy: 0.9967 - 13s/epoch - 7ms/step
#Epoch 20/20
#1875/1875 - 13s - loss: 0.0087 - accuracy: 0.9976 - 13s/epoch - 7ms/step
#Evaluate the model:
#313/313 - 1s - loss: 0.1353 - accuracy: 0.9822 - 1s/epoch - 3ms/step
##########################################################################
##### Epoch 20 and Optimizer Adagrad
# Epoch 20/20
# 1875/1875 - 10s - loss: 0.1814 - accuracy: 0.9496 - 10s/epoch - 6ms/step
# Evaluate the model:
# 313/313 - 1s - loss: 0.1801 - accuracy: 0.9481 - 839ms/epoch - 3ms/stepResult with epochs 20 and optimizer:
#######################################################################
#######Epoch 20 and optimizser RMSprop
#Epoch 19/20
#1875/1875 - 11s - loss: 0.0031 - accuracy: 0.9992 - 11s/epoch - 6ms/step
#Epoch 20/20
#1875/1875 - 11s - loss: 0.0023 - accuracy: 0.9993 - 11s/epoch - 6ms/step
#Evaluate the model:
#313/313 - 1s - loss: 0.1393 - accuracy: 0.9836 - 830ms/epoch - 3ms/step
########################################################################
######## hidden layer with size increased from 256 to 512 with RMSProp
# Epoch 19/20
# 1875/1875 - 16s - loss: 0.0010 - accuracy: 0.9998 - 16s/epoch - 9ms/step
# Epoch 20/20
# 1875/1875 - 16s - loss: 6.1600e-04 - accuracy: 0.9998 - 16s/epoch - 9ms/step
# Evaluate the model:
# 313/313 - 1s - loss: 0.1304 - accuracy: 0.9846 - 1s/epoch - 3ms/step


print (model.summary())