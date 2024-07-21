import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm #Representing a normal (Gaussian) continuous random variable. 
#It provides methods for probability density function (PDF), cumulative distribution function (CDF), and other statistics related to the normal distribution.
from datetime import datetime
import numpy as np #create a range of strike prices for plotting implied volatility
import pandas as pd #create a DataFrame that displays the input parameters for the Black-Scholes model in a structured tabular format. 
from pandas import DataFrame

# Created 2 main functions for the Black-Sholes Model
def BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    if Maturity_Time <= 0:
        raise ValueError("Time for maturity (Maturity_Time) must be positive (Must be above the cuurent year).")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    if Stock_Price <= 0 or Exercise_Price <= 0:
        raise ValueError("Underlying price (Stock_Price) and strike price (Exercise_Price) must be positive.")
    
    return (log(Stock_Price/Exercise_Price) + (risk_free + sigma**2 / 2.) * Maturity_Time) / (sigma * sqrt(Maturity_Time))

def BSM_function_2(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma) - sigma * sqrt(Maturity_Time)

def BSM_call(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return Stock_Price * norm.cdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) - Exercise_Price * exp(-risk_free * Maturity_Time) * norm.cdf(BSM_function_2(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma))

def BSM_put(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return Exercise_Price * exp(-risk_free * Maturity_Time) - Stock_Price + BSM_call(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)

#call functions to calculate the calls of the price function

def call_delta(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return norm.cdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma))

def call_gamma(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) / (Stock_Price * sigma * sqrt(Maturity_Time))

def call_vega(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return 0.01 * (Stock_Price * norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) * sqrt(Maturity_Time))

def call_theta(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return 0.01 * (-(Stock_Price * norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) * sigma) / (2 * sqrt(Maturity_Time)) - risk_free * Exercise_Price * exp(-risk_free * Maturity_Time) * norm.cdf(BSM_function_2(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)))

def call_rho(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return 0.01 * (Exercise_Price * Maturity_Time * exp(-risk_free * Maturity_Time) * norm.cdf(BSM_function_2(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)))

#Put functions to calculate the model price

def put_delta(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return -norm.cdf(-BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma))

def put_gamma(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) / (Stock_Price * sigma * sqrt(Maturity_Time))

def put_vega(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return 0.01 * (Stock_Price * norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) * sqrt(Maturity_Time))

def put_theta(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return 0.01 * (-(Stock_Price * norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) * sigma) / (2 * sqrt(Maturity_Time)) + risk_free * Exercise_Price * exp(-risk_free * Maturity_Time) * norm.cdf(-BSM_function_2(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)))

def put_rho(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma):
    return 0.01 * (-Exercise_Price * Maturity_Time * exp(-risk_free * Maturity_Time) * norm.cdf(-BSM_function_2(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)))

# Input values from user
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive number.")

Stock_Price = get_float_input("What is the current stock price? ")
Exercise_Price = get_float_input("What is the strike price? ")

while True:
    expiration_date = input("What is the expiration date of the options? Format: (mm-dd-yyyy) ")
    try:
        expiration_date = datetime.strptime(expiration_date, "%m-%d-%Y")
        break
    except ValueError as e:
        print(f"Error: {e}\nTry again.")
Maturity_Time = (expiration_date - datetime.utcnow()).days / 365

risk_free = get_float_input("What is the compounding risk-free interest rate in percent(%)? ") / 100
sigma = get_float_input("What is the volatility in percent(%)? ") / 100

data = {'Parameters': ['Stock_Price', 'Exercise_Price', 'Maturity_Time', 'risk_free', 'sigma'],
        'User-Input': [Stock_Price, Exercise_Price, Maturity_Time , risk_free , sigma]}
query = DataFrame(data, columns=['Parameters', 'User-Input'], 
                   index=['Underlying price', 'Strike price', 'Time to maturity', 'Risk-free interest rate', 'Volatility'])
print(query)

try:
    call_price = BSM_call(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)
    put_price = BSM_put(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)
    print(f"Call option price / Black Sholes Value: {call_price}")
    print(f"Put option price: {put_price}")
except ValueError as e:
    print(f"Error: {e}")

# Volatility Function
def volatility(option_type, Price, Stock_Price, Exercise_Price, Maturity_Time, risk_free, tol=1e-5, max_iterations=1000):
    sigma = 0.5  # initial guess
    for i in range(max_iterations):
        if option_type == 'C':
            price = BSM_call(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)
        elif option_type == 'P':
            price = BSM_put(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)
        vega = Stock_Price * norm.pdf(BSM_function_1(Stock_Price, Exercise_Price, Maturity_Time, risk_free, sigma)) * sqrt(Maturity_Time)
        price_diff = Price - price  # difference between market price and model price
        
        if abs(price_diff) < tol:
            return sigma
        
        sigma += price_diff / vega  # Newton-Raphson method to update sigma
    
    return None  # if it does not converge

option = input("Would you like to Put or Call option? Format: (P/C) ")
while option != 'P' and option != 'C':
    print("Error: This does not match the format (P/C). Try again.")
    option = input("Would you like to Put or Call option? (P/C) ")

Price = get_float_input("What is the option price? ")

implied_vol = volatility(option, Price, Stock_Price, Exercise_Price, Maturity_Time, risk_free)
if implied_vol is not None:
    print(f"The volatility is {implied_vol * 100:.2f} %.")
else:
    print("Cannot find right volatility.")

#Function to plot and represent the graph of the Black-Sholes Model
def plot_implied_volatility(Stock_Price, K_range, Maturity_Time, risk_free, option_type, Price):
    implied_vols = []
    for K in K_range:
        implied_vol = volatility(option_type, Price, Stock_Price, K, Maturity_Time, risk_free)
        implied_vols.append(implied_vol if implied_vol is not None else float('nan'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, implied_vols, marker='o')
    plt.title('Volatility vs. Strike Price')
    plt.xlabel('Strike Price (Exercise_Price)')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

# Define the range of strike prices
K_range = np.linspace(Stock_Price * 0.5, Stock_Price * 1.5, 10)

# Plot the implied volatility
plot_implied_volatility(Stock_Price, K_range, Maturity_Time, risk_free, option, Price)
