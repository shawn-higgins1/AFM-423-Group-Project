import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# To disable the gpu acceleration
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import kerastuner as kt

tf.get_logger().setLevel('ERROR')

#  Specify a seed if you want reproducibility
tf.random.set_seed(123)

# Make's for easier debugging / printing when displaying numpy arrays
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

PRINT_TENSORFLOW_PROGRESS = 0

VALIDATION_SPLIT = 0.25

DATA_DIR = "../data"
LOG_DIR = "./logs"


# Helper function for plotting the loss for the model
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def model_builder_1_layer(hp):
    model = keras.Sequential()

    hp_units = hp.Int('units', min_value=1, max_value=30, step=2)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.05, 1e-2, 1e-3])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
                  ])

    return model


def model_builder_2_layer(hp):
    model = keras.Sequential()

    hp_units_1 = hp.Int('first_units', min_value=1, max_value=15, step=2)
    model.add(keras.layers.Dense(units=hp_units_1, activation='relu'))
    hp_units_2 = hp.Int('second_units', min_value=1, max_value=15, step=2)
    model.add(keras.layers.Dense(units=hp_units_2, activation='relu'))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.05, 1e-2, 1e-3])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
                  ])

    return model


def train_and_test_model(option_type):
    # Load in the data
    raw_train = pd.read_csv(DATA_DIR + f"/train_{option_type}.csv", sep=',')
    raw_test1 = pd.read_csv(DATA_DIR + f"/test_{option_type}.csv", sep=',')
    raw_test2 = pd.read_csv(DATA_DIR + f"/{option_type}.csv", sep=',')

    train_dataset = raw_train.copy()
    test1_dataset = raw_test1.copy()
    test2_dataset = raw_test2.copy()

    train_features = train_dataset.copy()
    test1_features = test1_dataset.copy()
    test2_features = test2_dataset.copy()

    train_features.pop('C')
    train_features.pop('S')
    train_features.pop('k')

    test1_features.pop('C')
    test1_features.pop('S')
    test1_features.pop('k')

    test2_features.pop('C')
    test2_features.pop('S')
    test2_features.pop('k')
    test2_features.pop('Date')
    test2_features.pop('Volume')

    train_labels = train_features.pop('C/K')
    test1_labels = test1_features.pop('C/K')
    test2_labels = test2_features.pop('C/K')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model_checkpoint_dir = "./" + option_type
    model_log_dir = LOG_DIR + "/" + option_type

    tuner_1_layer = kt.Hyperband(model_builder_1_layer,
                                 objective='val_loss',
                                 max_epochs=15,
                                 factor=3,
                                 directory=model_checkpoint_dir + "/ANN",
                                 project_name='1_layer')

    tuner_1_layer.search(train_features, train_labels, epochs=50, validation_split=VALIDATION_SPLIT,
                         callbacks=[stop_early, tf.keras.callbacks.TensorBoard(model_log_dir + "/ANN/tuning/1_layer")],
                         verbose=PRINT_TENSORFLOW_PROGRESS)

    best_hps_1_layer = tuner_1_layer.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
        1 layer search complete. Optimal first layer units: {best_hps_1_layer.get('units')}, optimal learning rate: 
        {best_hps_1_layer.get('learning_rate')}.
        """)

    tuner_2_layer = kt.Hyperband(model_builder_2_layer,
                                 objective='val_loss',
                                 max_epochs=15,
                                 factor=3,
                                 directory=model_checkpoint_dir + "/ANN",
                                 project_name='2_layer')

    tuner_2_layer.search(train_features, train_labels, epochs=50, validation_split=VALIDATION_SPLIT,
                         callbacks=[stop_early, tf.keras.callbacks.TensorBoard(model_log_dir + "/ANN/tuning/2_layer")],
                         verbose=PRINT_TENSORFLOW_PROGRESS)

    best_hps_2_layer = tuner_2_layer.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
        2 layer optimized search complete. Optimal first layer units: {best_hps_2_layer.get('first_units')}, second: 
        {best_hps_2_layer.get('second_units')}, optimal learning rate: {best_hps_2_layer.get('learning_rate')}.
        """)

    layer1_model = tuner_1_layer.hypermodel.build(best_hps_1_layer)
    layer2_model = tuner_2_layer.hypermodel.build(best_hps_2_layer)

    start_time = perf_counter()
    history_1_layer = layer1_model.fit(train_features, train_labels, epochs=25, validation_split=VALIDATION_SPLIT,
                                       callbacks=[stop_early], verbose=PRINT_TENSORFLOW_PROGRESS)
    print("Elapsed time:", perf_counter() - start_time)

    start_time = perf_counter()
    history_2_layer = layer2_model.fit(train_features, train_labels, epochs=25, validation_split=VALIDATION_SPLIT,
                                       callbacks=[stop_early], verbose=PRINT_TENSORFLOW_PROGRESS)
    print("Elapsed time:", perf_counter() - start_time)

    if history_2_layer.history['val_loss'] > history_1_layer.history['val_loss']:
        history = history_1_layer
        hypermodel = tuner_1_layer.hypermodel.build(best_hps_1_layer)
        print("The 1 layer model had the lowest loss")
    else:
        history = history_2_layer
        hypermodel = tuner_2_layer.hypermodel.build(best_hps_2_layer)
        print("The 2 layer model had the lowest loss")

    plot_loss(history)

    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Retrain the model
    start_time = perf_counter()
    history = hypermodel.fit(train_features, train_labels, epochs=best_epoch,
                             callbacks=[tf.keras.callbacks.TensorBoard(model_log_dir + "/ANN/best")],
                             verbose=PRINT_TENSORFLOW_PROGRESS)
    print("Elapsed time:", perf_counter(), start_time)

    test_results = hypermodel.evaluate(test1_features, test1_labels)

    test_predictions = hypermodel.predict(test1_features).flatten()

    percentage_errors = []

    for i in range(len(test_predictions)):
        percentage_errors.append(np.abs(test1_labels[i] - test_predictions[i]) / test1_labels[i])

    plt.hist(percentage_errors, bins=25)
    plt.xlabel('Percentage Errors [C/K]')
    _ = plt.ylabel('Count')
    plt.show()

    print("Median Absolute Percentage Error: " + str(statistics.median(percentage_errors)))

    a = plt.axes(aspect='equal')
    plt.scatter(test1_labels, test_predictions)
    plt.xlabel('True Values [C/K]')
    plt.ylabel('Predictions [C/K]')
    lims = [0, 0.5]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test1_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [C/K]')
    _ = plt.ylabel('Count')
    plt.show()

    hist = pd.DataFrame(test_results).T
    hist = hist.rename(columns={0: "MSE", 1: "MAE", 2: "MAPE", 3: "R^2"})
    print(hist.tail())

    if len(test2_labels) > 0:
        test_results = hypermodel.evaluate(test2_features, test2_labels)

        hist = pd.DataFrame(test_results).T
        hist = hist.rename(columns={0: "MSE", 1: "MAE", 2: "MAPE", 3: "R^2"})
        print(hist.tail())

    hypermodel.save(f'saved_model/{option_type}_ann')


train_and_test_model("calls")
# train_and_test_model("puts")
