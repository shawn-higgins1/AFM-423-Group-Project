import numpy as np
from black_scholes import *
import scipy.stats as ss


# This script generates put and call option prices to be used instead of real data because historical options prices are
# either locked behind a paywall or there is insufficient volume for interesting analysis

def generate_options_data(n, call_data):
    # Initialize a matrix to hold the data
    output = np.empty((n + 1, 9), dtype=object)

    # Put the column headers in the first row of the matrix
    output[0] = np.array(
        [
            'C',
            'C/K',
            'D',
            'S',
            'S/K',
            'k',
            'r',
            'sigma',
            't'
        ]
    )

    for i in range(n):
        # Using distributions based on the test data generate random inputs for the black scholes equation
        index_price = np.round(ss.uniform.rvs(800, 400), 2)
        strike_price = index_price + 20 * ss.randint.rvs(-7, 8)
        time_to_expiry = np.round(ss.uniform.rvs(0.05, 1), 4)
        r = np.round(ss.uniform.rvs(0.01, 0.02), 4)
        volatility = np.round(ss.uniform.rvs(0.035, 0.20), 5)
        d = np.round(ss.uniform.rvs(0.020, 0.02), 3)
        expected_price = 0

        # Determine the expect black schole price for the option given the selected input parameters
        if call_data:
            expected_price = black_scholes_calls(
                index_price,
                d,
                time_to_expiry,
                strike_price,
                r,
                volatility
            )
        else:
            expected_price = black_scholes_puts(
                index_price,
                d,
                time_to_expiry,
                strike_price,
                r,
                volatility
            )

        # Round the data to sensible values
        expected_price = np.round(expected_price, 2)

        c_k = np.round(expected_price / strike_price, 5)
        s_k = np.round(index_price / strike_price, 5)

        # Add the selected inputs and the Black-Scholes price to the matrix
        output[i + 1] = np.array([
            expected_price,
            c_k,
            d,
            index_price,
            s_k,
            strike_price,
            r,
            volatility,
            time_to_expiry
        ])

    return output


# Parent directory where all the csv files are stored
DATA_DIR = "../data"

# Set a seed for reproducibility
np.random.seed(123)

# Generate the call train and test datasets
train_data = generate_options_data(50000, True)
test_data = generate_options_data(5000, True)

# Save the data
np.savetxt(DATA_DIR + "/train_calls.csv", train_data, delimiter=",", fmt='%s')
np.savetxt(DATA_DIR + "/test_calls.csv", test_data, delimiter=",", fmt='%s')

# Generate the put train and test datasets
train_data = generate_options_data(50000, False)
test_data = generate_options_data(5000, False)

# Save the data
np.savetxt(DATA_DIR + "/train_puts.csv", train_data, delimiter=",", fmt='%s')
np.savetxt(DATA_DIR + "/test_puts.csv", test_data, delimiter=",", fmt='%s')

