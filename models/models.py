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

# Make's for easier debugging / printing when displaying numpy arrays
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

PRINT_TENSORFLOW_PROGRESS = 0
PLOT_INDIVIDUAL_LOSSES = False

VALIDATION_SPLIT = 0.25

DATA_DIR = "../data"
LOG_DIR = "./logs"


class MyRSquared(tfa.metrics.RSquare):
    def __init__(self, name="r_squared", dtype=tf.float64, **kwargs):
        super().__init__(name=name, dtype=dtype, y_shape=(1,))


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

    hp_units = hp.Int('units', min_value=1, max_value=32, step=2)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.05, 1e-2, 1e-3])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      MyRSquared()
                  ])

    return model


def model_builder_2_layer(hp):
    model = keras.Sequential()

    hp_units_1 = hp.Int('first_units', min_value=1, max_value=16, step=2)
    model.add(keras.layers.Dense(units=hp_units_1, activation='relu'))
    hp_units_2 = hp.Int('second_units', min_value=1, max_value=16, step=2)
    model.add(keras.layers.Dense(units=hp_units_2, activation='relu'))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.05, 1e-2, 1e-3])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      MyRSquared()
                  ])

    return model


def train_and_test_model(model_type, option_type, train_features, train_labels, test_generated_features, test_generated_labels,
                         test_real_features, test_real_labels):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    models = []

    model_checkpoint_dir = f"./{option_type}"
    model_log_dir = LOG_DIR + f"./{option_type}"

    for i in range(len(train_labels)):
        print(f"Generating {option_type} {model_type} model {i}")

        tuner_1_layer = kt.Hyperband(model_builder_1_layer,
                                     objective='val_loss',
                                     max_epochs=15,
                                     factor=3,
                                     directory=model_checkpoint_dir + f"/{model_type}_{i}",
                                     project_name='1_layer')

        tuner_1_layer.search(train_features[i], train_labels[i], epochs=50, validation_split=VALIDATION_SPLIT,
                             callbacks=[stop_early,
                                        tf.keras.callbacks.TensorBoard(
                                            model_log_dir + f"/{model_type}_{i}/tuning/1_layer"
                                        )],
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
                                     directory=model_checkpoint_dir + f"/{model_type}_{i}",
                                     project_name='2_layer')

        tuner_2_layer.search(train_features[i], train_labels[i], epochs=50, validation_split=VALIDATION_SPLIT,
                             callbacks=[stop_early,
                                        tf.keras.callbacks.TensorBoard(
                                            model_log_dir + f"/{model_type}_{i}/tuning/2_layer")
                                        ],
                             verbose=PRINT_TENSORFLOW_PROGRESS)

        best_hps_2_layer = tuner_2_layer.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
            2 layer optimized search complete. Optimal first layer units: {best_hps_2_layer.get('first_units')}, second: 
            {best_hps_2_layer.get('second_units')}, optimal learning rate: {best_hps_2_layer.get('learning_rate')}.
            """)

        save_path = f'saved_model/{option_type}_{model_type}_{i}'

        if not os.path.exists(save_path) or len(os.listdir(save_path)) == 0:
            layer1_model = tuner_1_layer.hypermodel.build(best_hps_1_layer)
            layer2_model = tuner_2_layer.hypermodel.build(best_hps_2_layer)

            start_time = perf_counter()
            history_1_layer = layer1_model.fit(train_features[i], train_labels[i], epochs=50,
                                               validation_split=VALIDATION_SPLIT,
                                               callbacks=[stop_early], verbose=PRINT_TENSORFLOW_PROGRESS)
            print("Elapsed time:", perf_counter() - start_time)

            start_time = perf_counter()
            history_2_layer = layer2_model.fit(train_features[i], train_labels[i], epochs=50,
                                               validation_split=VALIDATION_SPLIT,
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

            if PLOT_INDIVIDUAL_LOSSES:
                plot_loss(history)

            val_loss_per_epoch = history.history['val_loss']
            best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch,))

            # Retrain the model
            start_time = perf_counter()
            hypermodel.fit(train_features[i], train_labels[i], epochs=best_epoch,
                           callbacks=[tf.keras.callbacks.TensorBoard(model_log_dir + f"/{model_type}_{i}/best")],
                           verbose=PRINT_TENSORFLOW_PROGRESS)
            print("Elapsed time:", perf_counter() - start_time)

            hypermodel.save(save_path)
            models.append(hypermodel)
        else:
            models.append(tf.keras.models.load_model(save_path, custom_objects={'MyRSquared': MyRSquared}))

    y_pred = np.array([])
    y_true = np.array([])

    for i in range(len(models)):
        test = models[i].predict(test_generated_features[i],
                                 verbose=PRINT_TENSORFLOW_PROGRESS).flatten()
        y_pred = np.concatenate((y_pred, test))
        y_true = np.concatenate((y_true, test_generated_labels[i].to_numpy()))

    percentage_errors = []

    for i in range(len(y_pred)):
        percentage_errors.append(np.abs(y_pred[i] - y_true[i]) / y_true[i])

    plt.hist(percentage_errors, bins=25)
    plt.xlabel('Percentage Errors [C/K]')
    _ = plt.ylabel('Count')
    plt.show()

    print("Median Absolute Percentage Error: " + str(statistics.median(percentage_errors)))

    a = plt.axes(aspect='equal')
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values [C/K]')
    plt.ylabel('Predictions [C/K]')
    lims = [0, 0.5]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = y_pred - y_true
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [C/K]')
    _ = plt.ylabel('Count')
    plt.show()

    mape = tf.keras.losses.MeanAbsolutePercentageError()
    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()
    r_squared = tfa.metrics.RSquare()

    print(f"""
    mape: {mape(y_true, y_pred).numpy()}, mse: {mse(y_true, y_pred).numpy()}, mae: {mae(y_true, y_pred).numpy()}, 
    r^2: {r_squared(y_true, y_pred).numpy()}
    """)

    y_pred = np.array([])
    y_true = np.array([])

    for i in range(len(models)):
        if len(test_real_labels[i]) > 0:
            y_pred = np.concatenate((y_pred,
                                     models[i].predict(test_real_features[i],
                                                       verbose=PRINT_TENSORFLOW_PROGRESS).flatten()), axis=0)
            y_true = np.concatenate((y_true, test_real_labels[i].to_numpy()))

    if len(y_true) > 0:
        print("Performance against the real test data:")

        percentage_errors = []

        for i in range(len(y_pred)):
            percentage_errors.append(np.abs(y_true[i] - y_pred[i]) / y_pred[i])

        print("Median Absolute Percentage Error: " + str(statistics.median(percentage_errors)))

        mape = tf.keras.losses.MeanAbsolutePercentageError()
        mae = tf.keras.losses.MeanAbsoluteError()
        mse = tf.keras.losses.MeanSquaredError()
        r_squared = tfa.metrics.RSquare()

        print(f"""
            mape: {mape(y_true, y_pred).numpy()}, mse: {mse(y_true, y_pred).numpy()}, mae: {mae(y_true, y_pred).numpy()}, 
            r^2: {r_squared(y_true, y_pred).numpy()}
            """)


def train_and_test_all_models(option_type):
    # Load in the data
    raw_train = pd.read_csv(DATA_DIR + f"/train_{option_type}.csv", sep=',')
    raw_test1 = pd.read_csv(DATA_DIR + f"/test_{option_type}.csv", sep=',')
    raw_test2 = pd.read_csv(DATA_DIR + f"/{option_type}.csv", sep=',')

    features = ['S/K', 't', 'D', 'r', 'sigma']

    train_dataset = raw_train.copy()
    test1_dataset = raw_test1.copy()
    test2_dataset = raw_test2.copy()

    train_features = train_dataset[features]
    test1_features = test1_dataset[features]
    test2_features = test2_dataset[features]

    train_labels = train_dataset['C/K']
    test1_labels = test1_dataset['C/K']
    test2_labels = test2_dataset['C/K']

    train_and_test_model("ANN", option_type, [train_features], [train_labels], [test1_features], [test1_labels],
                         [test2_features], [test2_labels])


print("Black Scholes Performance on the real test data for puts:")

real_data = pd.read_csv(DATA_DIR + "/puts.csv", sep=',')

y_true = real_data['C/K']
y_pred = real_data['black_scholes'] / real_data['k']

mape = tf.keras.losses.MeanAbsolutePercentageError()
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
r_squared = tfa.metrics.RSquare()

print(f"""
mape: {mape(y_true, y_pred).numpy()}, mse: {mse(y_true, y_pred).numpy()}, mae: {mae(y_true, y_pred).numpy()}, 
r^2: {r_squared(y_true, y_pred).numpy()}
""")

train_and_test_all_models("calls")
train_and_test_all_models("puts")
