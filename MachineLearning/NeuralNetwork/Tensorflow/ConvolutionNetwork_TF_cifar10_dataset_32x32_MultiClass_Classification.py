# reference
# https://www.youtube.com/watch?v=WAciKiDP2bo&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=4
# http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/
#
# image recognition and classification on cifar10 data set 32x32 RGB image
# using sequential model
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


# Use the below two lines incase need to use GPU
# physical_devices = tf.config.list_physical_devices
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("X_train shape ", x_train.shape)
print(y_train.shape)
print("###########################")
x_train[0]
print(x_train.dtype)

x_train = x_train.astype("float32")/255.0
y_test = y_test.astype("float32")/255.0
# verify the dataset by printing the first 5

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

"""plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i][0]])
    plt.show()
"""

# sequential API( very convenient , not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        # the input array will not be normalized in the beginning as we are using convolutional network
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128,3,activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
]
)

print(model.summary())

model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam( learning_rate = 3e-4),
        #optimizer=keras.optimizers.Adagrad(learning_rate=0.001),
        #optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["accuracy"],
    )

print ("Training the model:")
model.fit(x_train, y_train, batch_size =64, epochs = 10, verbose = 2)

print("Evaluate the model:")
model.evaluate(x_test, y_test, batch_size =64, verbose= 2)

#print (model.summary())

############################################################
#Epoch 10/10
#782/782 - 45s - loss: 0.8317 - accuracy: 0.7119 - 45s/epoch - 57ms/step
#Evaluate the model:
#313/313 - 3s - loss: 647.4875 - accuracy: 0.0846 - 3s/epoch - 11ms/step
###########################################################