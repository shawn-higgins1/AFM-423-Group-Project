import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt

#  Specify a seed if you want reproducabilty
tf.random.set_seed(123)

# Make's for easier debugging / printing when displaying numpy arrays
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)


# To disable the gpu acceleration
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Helper function for plotting the loss for the model
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


# Load in the data
DATA_DIR = "../data"

raw_train = pd.read_csv(DATA_DIR + "/train_calls.csv", sep=',')
raw_test = pd.read_csv(DATA_DIR + "/test_calls.csv", sep=',')

train_dataset = raw_train.copy()
test_dataset = raw_test.copy()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_features.pop('C')
train_features.pop('S')
train_features.pop('k')

test_features.pop('C')
test_features.pop('S')
test_features.pop('k')

train_labels = train_features.pop('C/K')
test_labels = test_features.pop('C/K')


def model_builder(hp):
    model = keras.Sequential()

    hp_units = hp.Int('units', min_value=1, max_value=15, step=1)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.05, 1e-2, 1e-3, 1e-4])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
                  ])

    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=15,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

start_time = perf_counter()
history = model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=1, epochs=20, callbacks=[stop_early])
print("Elapsed time:", perf_counter(), start_time)

plot_loss(history)

test_results = model.evaluate(
    test_features, test_labels,
    verbose=0)

test_predictions = model.predict(test_features).flatten()

precentage_errors = []

for i in range(len(test_predictions)):
    precentage_errors.append(np.abs(test_labels[i] - test_predictions[i]) / test_labels[i])

plt.hist(precentage_errors, bins=25)
plt.xlabel('Percentage Errors [C/K]')
_ = plt.ylabel('Count')
plt.show()

print("Median Absolute Percentage Error: " + str(statistics.median(precentage_errors)))

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [C/K]')
plt.ylabel('Predictions [C/K]')
lims = [0, 0.5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [C/K]')
_ = plt.ylabel('Count')
plt.show()

hist = pd.DataFrame(test_results).T
hist = hist.rename(columns={0: "MSE", 1: "MAE", 2: "MAPE", 3: "R^2"})
print(hist.tail())
