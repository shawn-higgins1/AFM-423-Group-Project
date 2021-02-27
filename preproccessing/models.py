import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from time import perf_counter

from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(123)

np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

# To disable the gpu acceleration
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


model = keras.Sequential([
    layers.Dense(5, activation='relu'),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=[
                  "mean_absolute_error",
                  "mean_absolute_percentage_error",
                  tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
              ]
)

start_time = perf_counter()
history = model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=20)
print("Elapsed time:", perf_counter(), start_time)

plot_loss(history)

test_results = model.evaluate(
    test_features, test_labels,
    verbose=0)

test_predictions = model.predict(test_features).flatten()

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
print(hist.tail())
