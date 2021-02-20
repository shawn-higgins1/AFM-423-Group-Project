import pandas as pd
from black_scholes import black_scholes_with_dividends

# Parent directory where all the csv files are stored
DATA_DIR = "../data"

# Source: https://www.m-x.ca/nego_fin_jour_en.php?symbol=SXO&from=2020-03-02&to=2020-04-01#cotes
# Load in the data for the S&P TSX 60 options
march_option_prices = pd.read_csv(DATA_DIR + '/quote_SXO_20200101_20200430.csv', sep=',')[[
    'Date',
    'Strike Price',
    'Expiry Date',
    'Call/Put',
    'Settlement Price',
    'Volume',
    'Implied Volatility'
]]

# Cleanup the options data removing options that had no volume traded during the day
march_option_prices = march_option_prices[march_option_prices["Volume"] > 0]
march_option_prices["Date"] = pd.to_datetime(march_option_prices["Date"])
march_option_prices["Expiry Date"] = pd.to_datetime(march_option_prices["Expiry Date"])

# Source: https://www.spglobal.com/spdji/en/indices/equity/sp-tsx-60-index/#data
# Load in the historical prices for the S&P TSX 60
sp_60_prices = pd.read_excel(DATA_DIR + '/PerformanceGraphExport.xls')

# Join the option prices and the index prices
march_option_prices = pd.merge(march_option_prices, sp_60_prices, left_on='Date', right_on='Effective date')

# Cleanup the columns after the join
march_option_prices = march_option_prices.rename(columns={'S&P/TSX 60 Index': 'Index Price'})
march_option_prices = march_option_prices.drop(
    ['Effective date', 'Continously Compounded Annual Return', 'Annual Return'], axis=1
)

# Source: https://ycharts.com/companies/XIU.TO/dividend_yield
# Load in the data for the annual dividend yield for the S&P TSX 60
dividend_yield = pd.read_excel(DATA_DIR + '/Dividend Yield.xlsx')
dividend_yield = dividend_yield.rename(columns={'Value': 'Dividend Yield'})

march_option_prices = pd.merge(march_option_prices, dividend_yield, on='Date', how='left')

# Source: https://www.bankofcanada.ca/rates/interest-rates/t-bill-yields/#tbills
# Load in the continuously compounded risk free interest rate
interest_rates = pd.read_csv(DATA_DIR + '/1-month-treasury-rates.csv')
interest_rates["Date"] = pd.to_datetime(interest_rates["Date"])

# Calculate the t value for the options ( the amount of time in years until the option expires)
march_option_prices = pd.merge(march_option_prices, interest_rates, on='Date', how='left')
march_option_prices["Time Delta"] = (march_option_prices["Expiry Date"] - march_option_prices["Date"])
march_option_prices["t"] = march_option_prices["Time Delta"].dt.days / 366
march_option_prices = march_option_prices.drop(["Time Delta", "Expiry Date"], axis=1)

# Split the options into calls and puts
call_options = march_option_prices[march_option_prices["Call/Put"] == 0]
call_options = call_options.drop(["Call/Put"], axis=1)
call_options['Black Scholes'] = 0.0

# Calculate the black scholes price for the calls
for i, call_option in call_options.iterrows():
    call_options.at[i, 'Black Scholes'] = black_scholes_with_dividends(
        call_option['Index Price'],
        call_option['Dividend Yield'],
        call_option['t'],
        call_option['Strike Price'],
        call_option['Interest Rate'],
        call_option['Volatility']
    )

# Save the data in a csv
call_options.to_csv(DATA_DIR + '/calls.csv')

put_options = march_option_prices[march_option_prices["Call/Put"] == 1]
put_options = put_options.drop(["Call/Put"], axis=1)
put_options['Black Scholes'] = 0.0

# Calculate the black scholes price for the puts
for i, put_option in put_options.iterrows():
    put_options.at[i, 'Black Scholes'] = black_scholes_with_dividends(
        put_option['Index Price'],
        put_option['Dividend Yield'],
        put_option['t'],
        put_option['Strike Price'],
        put_option['Interest Rate'],
        put_option['Volatility']
    )

# Save the data in a csv
put_options.to_csv(DATA_DIR + '/puts.csv')
