import numpy as np
from black_scholes import *
import scipy.stats as ss


def generate_options_data(n, call_data):
    output = np.empty((n + 1, 9), dtype=object)
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
        index_price = np.round(ss.uniform.rvs(800, 400), 2)
        strike_price = index_price + 20 * ss.randint.rvs(-5, 6)
        time_to_expiry = np.round(ss.lognorm.rvs(1.5, 0.05, 0.15), 4)
        r = np.round(ss.uniform.rvs(0.01, 0.02), 4)
        volatility = np.round(ss.lognorm.rvs(0.75, 0, 0.08), 5)
        d = np.round(ss.uniform.rvs(0.020, 0.015), 3)
        expected_price = 0

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

        expected_price = np.round(expected_price, 2)

        c_k = np.round(expected_price / strike_price, 5)
        s_k = np.round(index_price / strike_price, 5)

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

np.random.seed(123)

train_data = generate_options_data(10000, True)
test_data = generate_options_data(1000, True)

np.savetxt(DATA_DIR + "/train_calls.csv", train_data, delimiter=",", fmt='%s')
np.savetxt(DATA_DIR + "/test_calls.csv", test_data, delimiter=",", fmt='%s')

train_data = generate_options_data(10000, False)
test_data = generate_options_data(1000, False)

np.savetxt(DATA_DIR + "/train_puts.csv", train_data, delimiter=",", fmt='%s')
np.savetxt(DATA_DIR + "/test_puts.csv", test_data, delimiter=",", fmt='%s')

