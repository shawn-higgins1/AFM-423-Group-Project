import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# To disable the gpu acceleration (not recommended)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import kerastuner as kt

# Hide the verbose tensorflow debugging messages
tf.get_logger().setLevel('ERROR')

# Make's for easier debugging / printing when displaying numpy arrays
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

# Constants used to control what the program outputs
PRINT_TENSORFLOW_PROGRESS = 0
SHOW_PLOTS = False

VALIDATION_SPLIT = 0.20

# Constants for the directories to find the data and save the logs
DATA_DIR = "../data"
LOG_DIR = "./logs"


# Custom RSquared Metric used to evaluate the R squared value for a model
class MyRSquared(tfa.metrics.RSquare):
    def __init__(self, name="r_squared", dtype=tf.float64, **kwargs):
        super().__init__(name=name, dtype=dtype, y_shape=(1,))


# Helper function for plotting the loss for the model
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title("Train and Validation Loss Per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


# This function builds a 1 layer Feed Forward Neural network that is used by the Hyperband tuner to find the optimal
# hyper-parameters. The hp.Choice values represent the hyper-parameters that will get tested
def model_builder_1_layer(hp):
    model = keras.Sequential()

    hp_units = hp.Choice('units', values=[4, 12, 16, 24, 32, 48, 64])
    model.add(keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 1e-2, 1e-3, 1e-4])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      MyRSquared()
                  ])

    return model


# Similar to model_builder_1_layer expect this builds a 2-layer Feed Forward Neural Network
def model_builder_2_layer(hp):
    model = keras.Sequential()

    hp_units_1 = hp.Choice('first_units', values=[4, 12, 16, 24, 32])
    model.add(keras.layers.Dense(units=hp_units_1, activation="relu"))
    hp_units_2 = hp.Choice('second_units', values=[4, 12, 16, 24, 32])
    model.add(keras.layers.Dense(units=hp_units_2, activation="relu"))
    model.add(keras.layers.Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 1e-2, 1e-3, 1e-4])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      "mean_absolute_error",
                      "mean_absolute_percentage_error",
                      MyRSquared()
                  ])

    return model


# Helper function that determines the 4 metrics we use to evaluate each model based on y_pred and y_true
def calc_performance(y_pred, y_true):
    # Load the metrics from Tensorflow
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    mse = tf.keras.metrics.MeanSquaredError()
    r_squared = tfa.metrics.RSquare()

    # Output the metrics
    print(f"""
    mape: {mape(y_true, y_pred).numpy()}, mse: {mse(y_true, y_pred).numpy()}, mae: {mae(y_true, y_pred).numpy()}, 
    r^2: {r_squared(y_true, y_pred).numpy()}
    """)


# This is the main drive function. First it performs a Hyperband search to find the optimal model hyper-parameters and
# then in retrains the optimal model on all of the data and computes the performance metrics for the model
def train_and_test_model(model_type, option_type, train_features, train_labels, test_generated_features,
                         test_generated_labels,
                         test_real_features, test_real_labels):
    models = []

    model_checkpoint_dir = f"./{option_type}"
    model_log_dir = LOG_DIR + f"./{option_type}"
    total_time = 0.0

    # In the case of the MNN the train and test data is broken up by the criteria for each module in the MNN
    for i in range(len(train_labels)):
        print(f"Generating {option_type} {model_type} model {i}. Total train: {len(train_labels[i])} total test: " +
              f"{len(test_generated_labels[i])}")

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Find the optimal 1 layer model. If there is pre-existing checkpoints in the folder specified by directory
        # then the results of that search are returned
        tuner_1_layer = kt.Hyperband(model_builder_1_layer,
                                     objective='val_loss',
                                     max_epochs=25,
                                     factor=3,
                                     directory=model_checkpoint_dir + f"/{model_type}_{i}",
                                     project_name='1_layer')

        tuner_1_layer.search(train_features[i], train_labels[i], epochs=100, validation_split=VALIDATION_SPLIT,
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

        # Find the optimal 2 layer model
        tuner_2_layer = kt.Hyperband(model_builder_2_layer,
                                     objective='val_loss',
                                     max_epochs=25,
                                     factor=3,
                                     directory=model_checkpoint_dir + f"/{model_type}_{i}",
                                     project_name='2_layer')

        tuner_2_layer.search(train_features[i], train_labels[i], epochs=100, validation_split=VALIDATION_SPLIT,
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

        # If there is a saved copy of the best model than load the saved model otherwise, train a new model.
        if not os.path.exists(save_path) or len(os.listdir(save_path)) == 0:
            # Build and train the optimal 1 layer and 2 layer models
            layer1_model = tuner_1_layer.hypermodel.build(best_hps_1_layer)
            layer2_model = tuner_2_layer.hypermodel.build(best_hps_2_layer)

            start_time = perf_counter()
            history_1_layer = layer1_model.fit(train_features[i], train_labels[i], epochs=200,
                                               validation_split=VALIDATION_SPLIT,
                                               verbose=PRINT_TENSORFLOW_PROGRESS,
                                               callbacks=[
                                                   tf.keras.callbacks.TensorBoard(
                                                       model_log_dir + f"/{model_type}_{i}/final/1_layer"
                                                   )]
                                               )
            model_1_time = perf_counter() - start_time

            start_time = perf_counter()
            history_2_layer = layer2_model.fit(train_features[i], train_labels[i], epochs=200,
                                               validation_split=VALIDATION_SPLIT,
                                               verbose=PRINT_TENSORFLOW_PROGRESS,
                                               callbacks=[
                                                   tf.keras.callbacks.TensorBoard(
                                                       model_log_dir + f"/{model_type}_{i}/final/2_layer"
                                                   )]
                                               )
            model_2_time = perf_counter() - start_time

            # Determine the average validation loss for both models over the final 20 epochs. This average prevents
            # a spike in the validation loss on the last epoch from skewing which model we end up selecting.
            layer_1_loss = np.average(history_1_layer.history['val_loss'][-20:])
            layer_2_loss = np.average(history_2_layer.history['val_loss'][-20:])

            # Choose the best model
            if layer_2_loss > layer_1_loss:
                hypermodel = layer1_model
                print("The 1 layer model had the lowest loss")
                total_time += model_1_time

                if SHOW_PLOTS:
                    plot_loss(history_1_layer)
            else:
                hypermodel = layer2_model
                print("The 2 layer model had the lowest loss")
                total_time += model_2_time

                if SHOW_PLOTS:
                    plot_loss(history_2_layer)

            # Save the best model
            hypermodel.save(save_path)

            # For an MNN append the model to the list of the over sub models.
            models.append(hypermodel)
        else:
            model = tf.keras.models.load_model(save_path, custom_objects={'MyRSquared': MyRSquared})
            models.append(model)
            print(model.summary())

    print("Took", total_time, "seconds to train the complete model")

    # Evaluate the model(s) based on the generated data
    y_pred = np.array([])
    y_true = np.array([])

    for i in range(len(models)):
        # Generate the prediction for the test data. Since the NN outputs C / K instead of just C. We multiply the
        # predictions by K to faciliate a better comparison with Black-Scholes
        predictions = models[i].predict(test_generated_features[i],
                                        verbose=PRINT_TENSORFLOW_PROGRESS).flatten() * \
                      test_generated_labels[i]['k'].to_numpy()

        # Since we know that an option with never have a negative price run a max function over the prediction prices
        maxer = lambda x: max(x, 0)
        vector_maxer = np.vectorize(maxer)

        predictions = vector_maxer(predictions)

        # Calculate the performance for the individual model
        print(f"Individual performance for the model {i} on the generated data:")
        calc_performance(predictions, test_generated_labels[i]['C'].to_numpy())

        # In the base of a MNN concatenate the predictions together so we can evaluate the predictions across all the
        # modules
        y_pred = np.concatenate((y_pred, predictions))
        y_true = np.concatenate((y_true, test_generated_labels[i]['C'].to_numpy()))

    percentage_errors = []

    # Calculate the Median absolute percentage error. This is valuable because for some options with very low prices the
    # MAPE can get skewed because even small (1 or 2 cent) differences in prices can result in a huge bsolute Percentage
    # error
    for i in range(len(y_pred)):
        if y_true[i] != 0:
            percentage_errors.append(np.abs(y_pred[i] - y_true[i]) / y_true[i])
        else:
            percentage_errors.append(np.abs(y_pred[i] - y_true[i]) / 0.001)

    print("Median Absolute Percentage Error: " + str(statistics.median(percentage_errors)))

    if SHOW_PLOTS:
        # plt.hist(percentage_errors, bins=25)
        # plt.xlabel('Percentage Errors [C/K]')
        # _ = plt.ylabel('Count')
        # plt.show()

        error = y_pred - y_true
        plt.hist(error, bins=25)
        plt.title("Histogram of the Predictions Errors")
        plt.xlabel('Prediction Error [C]')
        _ = plt.ylabel('Count')
        plt.show()

    # Calculate the performance across the combined predictions from each model
    calc_performance(y_pred, y_true)

    # Perform the same testing procedure on the real data
    y_pred = np.array([])
    y_true = np.array([])

    for i in range(len(models)):
        if len(test_real_labels[i]) > 0:
            predictions = models[i].predict(test_real_features[i],
                                            verbose=PRINT_TENSORFLOW_PROGRESS).flatten() * \
                          test_real_labels[i]['k'].to_numpy()

            maxer = lambda x: max(x, 0)
            vector_maxer = np.vectorize(maxer)

            predictions = vector_maxer(predictions)

            print(f"Individual performance for the model {i} on the real data:")
            calc_performance(predictions, test_real_labels[i]['C'].to_numpy())

            y_pred = np.concatenate((y_pred, predictions), axis=0)
            y_true = np.concatenate((y_true, test_real_labels[i]['C'].to_numpy()))

    if len(y_true) > 0:
        print("Performance against the real test data:")

        percentage_errors = []

        for i in range(len(y_pred)):
            if y_true[i] != 0:
                percentage_errors.append(np.abs(y_pred[i] - y_true[i]) / y_true[i])
            else:
                percentage_errors.append(np.abs(y_pred[i] - y_true[i]) / 0.001)

        print("Median Absolute Percentage Error: " + str(statistics.median(percentage_errors)))

        calc_performance(y_pred, y_true)

        if SHOW_PLOTS:
            a = plt.axes(aspect='equal')
            plt.scatter(y_true, y_pred)
            plt.xlabel('True Values [C]')
            plt.ylabel('Predictions [C]')
            plt.title("Real Option Prices vs. Predicted Option Prices")
            lims = [0, 350]
            plt.xlim(lims)
            plt.ylim(lims)
            _ = plt.plot(lims, lims)
            plt.show()


# This function runs the three distinct model we wish to test ANN, MNN with 3 modules and MNN with 9 models for the
# specified options_type
def train_and_test_all_models(option_type):
    # Load in the data
    raw_train = pd.read_csv(DATA_DIR + f"/train_{option_type}.csv", sep=',')
    raw_test_generated = pd.read_csv(DATA_DIR + f"/test_{option_type}.csv", sep=',')
    raw_test_real = pd.read_csv(DATA_DIR + f"/{option_type}.csv", sep=',')

    features = ['S/K', 't', 'D', 'r', 'sigma']

    train_dataset = raw_train.copy()
    test_dataset_generated = raw_test_generated.copy()
    test_dataset_real = raw_test_real.copy()

    # Split the data into features and the response
    train_features = train_dataset[features]
    test_features_generated = test_dataset_generated[features]
    test_features_real = test_dataset_real[features]

    train_labels = train_dataset['C/K']
    test_labels_generated = test_dataset_generated[['C', 'k']]
    test_labels_real = test_dataset_real[['C', 'k']]

    # Find the best ANN model and the test its performance against the test data
    train_and_test_model("ANN", option_type, [train_features], [train_labels], [test_features_generated],
                         [test_labels_generated], [test_features_real], [test_labels_real])

    # Split the data into three distinct groups for the MNN is 3 modules
    if option_type == 'calls':
        split_funcs = [
            lambda x: x['S/K'] > 1.05,
            lambda x: x['S/K'] < 0.97,
            lambda x: 0.97 <= x['S/K'] <= 1.05,
        ]
    else:
        split_funcs = [
            lambda x: x['S/K'] > 1.03,
            lambda x: x['S/K'] < 0.95,
            lambda x: 0.95 <= x['S/K'] <= 1.03,
        ]
    mnn1_train_features = []
    mnn1_train_labels = []

    mnn1_test_features_generated = []
    mnn1_test_labels_generated = []

    mnn1_test_features_real = []
    mnn1_test_labels_real = []

    for split_func in split_funcs:
        # Split the training data
        filtered_dataset = train_dataset[train_dataset.apply(split_func, axis=1)]

        mnn1_train_features.append(filtered_dataset[features])
        mnn1_train_labels.append(filtered_dataset['C/K'])

        # Split the test data
        filtered_dataset = test_dataset_generated[test_dataset_generated.apply(split_func, axis=1)]

        mnn1_test_features_generated.append(filtered_dataset[features])
        mnn1_test_labels_generated.append(filtered_dataset[['C', 'k']])

        # Split the real data
        filtered_dataset = test_dataset_real[test_dataset_real.apply(split_func, axis=1)]

        mnn1_test_features_real.append(filtered_dataset[features])
        mnn1_test_labels_real.append(filtered_dataset[['C', 'k']])

    # Find the best MNN with 3 sub models and evaluate it
    train_and_test_model("MNN1", option_type, mnn1_train_features, mnn1_train_labels, mnn1_test_features_generated,
                         mnn1_test_labels_generated, mnn1_test_features_real, mnn1_test_labels_real)

    # Split the data into nine distinct groups for the MNN is 9 sub-models
    if option_type == 'calls':
        split_funcs = [
            lambda x: x['S/K'] > 1.05 and x['t'] < 0.06,
            lambda x: x['S/K'] < 0.97 and x['t'] < 0.06,
            lambda x: 0.97 <= x['S/K'] <= 1.05 and x['t'] < 0.06,
            lambda x: x['S/K'] > 1.05 and x['t'] > 0.1,
            lambda x: x['S/K'] < 0.97 and x['t'] > 0.1,
            lambda x: 0.97 <= x['S/K'] <= 1.05 and x['t'] > 0.1,
            lambda x: x['S/K'] > 1.05 and 0.06 <= x['t'] <= 0.1,
            lambda x: x['S/K'] < 0.97 and 0.06 <= x['t'] <= 0.1,
            lambda x: 0.97 <= x['S/K'] <= 1.05 and 0.06 <= x['t'] <= 0.1,
        ]
    else:
        split_funcs = [
            lambda x: x['S/K'] > 1.03 and x['t'] < 0.06,
            lambda x: x['S/K'] < 0.95 and x['t'] < 0.06,
            lambda x: 0.95 <= x['S/K'] <= 1.03 and x['t'] < 0.06,
            lambda x: x['S/K'] > 1.03 and x['t'] > 0.1,
            lambda x: x['S/K'] < 0.95 and x['t'] > 0.1,
            lambda x: 0.95 <= x['S/K'] <= 1.03 and x['t'] > 0.1,
            lambda x: x['S/K'] > 1.03 and 0.06 <= x['t'] <= 0.1,
            lambda x: x['S/K'] < 0.95 and 0.06 <= x['t'] <= 0.1,
            lambda x: 0.95 <= x['S/K'] <= 1.03 and 0.06 <= x['t'] <= 0.1,
        ]
    mnn2_train_features = []
    mnn2_train_labels = []

    mnn2_test_features_generated = []
    mnn2_test_labels_generated = []

    mnn2_test_features_real = []
    mnn2_test_labels_real = []

    for split_func in split_funcs:
        # Split the training dataset
        filtered_dataset = train_dataset[train_dataset.apply(split_func, axis=1)]

        mnn2_train_features.append(filtered_dataset[features])
        mnn2_train_labels.append(filtered_dataset['C/K'])

        # Split the test data
        filtered_dataset = test_dataset_generated[test_dataset_generated.apply(split_func, axis=1)]

        mnn2_test_features_generated.append(filtered_dataset[features])
        mnn2_test_labels_generated.append(filtered_dataset[['C', 'k']])

        # Split the real data
        filtered_dataset = test_dataset_real[test_dataset_real.apply(split_func, axis=1)]

        mnn2_test_features_real.append(filtered_dataset[features])
        mnn2_test_labels_real.append(filtered_dataset[['C', 'k']])

    # Find the best MNN with 9 sub models and evaluate it
    train_and_test_model("MNN2", option_type, mnn2_train_features, mnn2_train_labels, mnn2_test_features_generated,
                         mnn2_test_labels_generated, mnn2_test_features_real, mnn2_test_labels_real)


# Read in the real puts data
print("Black Scholes Performance on the real test data for puts:")
real_data = pd.read_csv(DATA_DIR + "/puts.csv", sep=',')

# Evaluate the performance of Black-Scholes on the real data
y_true = real_data['C']
y_pred = real_data['black_scholes']

if SHOW_PLOTS:
    a = plt.axes(aspect='equal')
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values [C]')
    plt.ylabel('Predictions [C]')
    plt.title("Real Option Prices vs. Predicted Option Prices")
    lims = [0, 350]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

mape = tf.keras.losses.MeanAbsolutePercentageError()
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
r_squared = tfa.metrics.RSquare()

print(f"""
mape: {mape(y_true, y_pred).numpy()}, mse: {mse(y_true, y_pred).numpy()}, mae: {mae(y_true, y_pred).numpy()}, 
r^2: {r_squared(y_true, y_pred).numpy()}
""")

# Train and evaluate every model type for both calls and puts
train_and_test_all_models("puts")
train_and_test_all_models("calls")
