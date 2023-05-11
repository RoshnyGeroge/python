# Fuel efficiency Prediction using AutoMPG data based available UCI with TF
# uses 3 approaches  single variable linear regress, multivariable linear regression and with DNN.

# Use seaborn for pairplot.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

np.set_printoptions(precision=3, suppress=True)

#get data

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,na_values='?', comment='\t', sep=' ', skipinitialspace=True)

dataset=raw_dataset.copy()
print(dataset.tail())

#Clean the data
print("Unknown values summary:")
print(dataset.isna().sum())

dataset = dataset.dropna()# remove rows with  unknown values
print("Shape dataset :", dataset.shape)

#replace Origin with LoV
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')# create boolen fields for Origin category values
print("dataset : ",dataset.tail())

print('dataset col types:\n ########\n', dataset.dtypes)

#dataset['Cylinders']= dataset['Cylinders'].astype(float)
#dataset['Model Year']= dataset['Model Year'].astype(float)
dataset = dataset.astype('float32') # convert datatype of all colms to float32
print('dataset col types:\n ########\n', dataset.dtypes)

# Split the data to training and test data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#Inspect data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
print("Describe data set:")
print(train_dataset.describe().transpose())

# seperate the target label
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


"""
# Linear Regression
# start with one variable , say Horse power to predict MPG

horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)# calculate mean and varience

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#Training the model
#%%time
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # o for Suppress logging.
    verbose=2,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# Visualize model training
#%%time
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=2)
######save Results###########
#Epoch 100/100
#8/8 - 0s - loss: 3.8059 - val_loss: 4.1647 - 33ms/epoch - 4ms/step
#3/3 - 0s - loss: 3.6531 - 22ms/epoch - 7ms/step
###############################
"""
#Multi variable normalization

#Normalize  the features
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))# calculates mean across all axes and varience
print(normalizer.mean.numpy())

""""
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.summary()

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#Training the model
#%%time
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # o for Suppress logging.
    verbose=2,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

linear_model.summary()

# Visualize model training
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

test_results = {}

test_results['linear_model'] = linear_model.evaluate(
    test_features,
    test_labels, verbose=2)

plot_loss(history)
"""
###########Test Result######
#Epoch 99/100
#8/8 - 0s - loss: 2.4756 - val_loss: 2.4759 - 44ms/epoch - 5ms/step
#Epoch 100/100
#8/8 - 0s - loss: 2.4807 - val_loss: 2.5111 - 44ms/epoch - 5ms/step
# 3/3 - 0s - loss: 2.5184 - 23ms/epoch - 8ms/step
####################################
#Regression with a deep neural network (DNN)

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(units=1)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error')

#Training the model
#%%time
history = model.fit(
    train_features,
    train_labels,
    epochs=100,
    # o for Suppress logging.
    verbose=2,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

model.summary()

# Visualize model training
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

test_results = {}

test_results['model'] = model.evaluate(
    test_features,
    test_labels, verbose=2)

plot_loss(history)
