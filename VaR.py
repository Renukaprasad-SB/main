# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.stats import norm

# Define the time period for historical data
years_of_data = 10  # Number of years of historical data
end_date = dt.datetime.now().date()  # Today's date
start_date = end_date - dt.timedelta(days=years_of_data * 365)  # Start date for historical data

# Define the portfolio tickers (assets in the portfolio)
portfolio_tickers = ["NIFTYBEES.NS", "ASIANPAINT.NS", "TECHM.NS", "SBIN.NS", "PIDILITIND.NS"]

# Fetch historical closing prices for the defined tickers
closing_prices_df = pd.DataFrame()
for ticker in portfolio_tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    closing_prices_df[ticker] = stock_data["Close"]

# Handle missing data
closing_prices_df.dropna(axis=1, inplace=True)  # Remove stocks with missing data

# Calculate daily log returns
log_returns_df = np.log(closing_prices_df / closing_prices_df.shift(1))
log_returns_df.dropna(inplace=True)  # Remove any NaN values resulting from the shift

# Define portfolio details
initial_portfolio_value = 10_000_000  # Portfolio value in INR (10 million)
portfolio_weights = np.array([0.17, 0.28, 0.44, 0.08, 0.03])[:closing_prices_df.shape[1]]

# Calculate portfolio historical returns
portfolio_daily_returns = (log_returns_df * portfolio_weights).sum(axis=1)

# Define multiple confidence levels
confidence_levels = [0.90, 0.95, 0.99]

# Get user input for time horizon (in days)
time_horizon = int(input("Enter the time horizon for VaR calculation (in days): "))

# 🔴 Historical VaR Calculation
historical_VaR_results = {}
for confidence_level in confidence_levels:
    historical_VaR = -np.percentile(portfolio_daily_returns, (1 - confidence_level) * 100) * np.sqrt(time_horizon) * initial_portfolio_value
    historical_VaR_results[confidence_level] = historical_VaR

# 🔵 Parametric VaR Calculation
cov_matrix = log_returns_df.cov()
portfolio_std_dev = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights)))
parametric_VaR_results = {}

for confidence_level in confidence_levels:
    z_score = norm.ppf(confidence_level)
    parametric_VaR = z_score * portfolio_std_dev * np.sqrt(time_horizon) * initial_portfolio_value
    parametric_VaR_results[confidence_level] = parametric_VaR

# 🔶 Monte Carlo Simulation for VaR
num_simulations = 10_000  # Number of Monte Carlo simulations
mean_return = portfolio_daily_returns.mean()
std_dev_return = portfolio_daily_returns.std()

# Generate random returns using normal distribution
simulated_returns = np.random.normal(mean_return, std_dev_return, (num_simulations, time_horizon))
simulated_portfolio_returns = simulated_returns.sum(axis=1)
simulated_portfolio_values = initial_portfolio_value * (1 + simulated_portfolio_returns)
simulated_losses = initial_portfolio_value - simulated_portfolio_values  # Calculate losses

# Monte Carlo VaR Calculation
monte_carlo_VaR_results = {}
for confidence_level in confidence_levels:
    mc_VaR = -np.percentile(simulated_losses, (1 - confidence_level) * 100)
    monte_carlo_VaR_results[confidence_level] = mc_VaR

# 🔥 Display VaR Results
print("\n🔴 Historical VaR Results:")
for level, value in historical_VaR_results.items():
    print(f" - {int(level * 100)}%: INR {value:,.2f}")

print("\n🔵 Parametric VaR Results:")
for level, value in parametric_VaR_results.items():
    print(f" - {int(level * 100)}%: INR {value:,.2f}")

print("\n🟢 Monte Carlo VaR Results:")
for level, value in monte_carlo_VaR_results.items():
    print(f" - {int(level * 100)}%: INR {value:,.2f}")