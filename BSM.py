import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import optimize

# Constants
DAYS_PER_YEAR = 365.0

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize the Black-Scholes Model.
        
        :param S: Current stock price
        :param K: Strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate (as a decimal)
        :param sigma: Volatility (as a decimal)
        """
        self.validate_inputs(S, K, T, r, sigma)
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    @staticmethod
    def validate_inputs(S, K, T, r, sigma):
        """Validate input parameters."""
        if S <= 0 or K <= 0:
            raise ValueError("Stock price and strike price must be positive.")
        if T <= 0:
            raise ValueError("Time to maturity must be positive.")
        if r < 0 or r > 1:
            raise ValueError("Risk-free rate should be between 0 and 1.")
        if sigma <= 0 or sigma > 1:
            raise ValueError("Volatility should be between 0 and 1.")

    def d1(self):
        """Calculate d1 parameter."""
        return (log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * sqrt(self.T))

    def d2(self):
        """Calculate d2 parameter."""
        return self.d1() - self.sigma * sqrt(self.T)

    def call_price(self):
        """Calculate call option price."""
        return self.S * norm.cdf(self.d1()) - self.K * exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        """Calculate put option price."""
        return self.K * exp(-self.r * self.T) - self.S + self.call_price()

    def call_delta(self):
        """Calculate call option delta."""
        return norm.cdf(self.d1())

    def put_delta(self):
        """Calculate put option delta."""
        return -norm.cdf(-self.d1())

    def gamma(self):
        """Calculate gamma (same for call and put)."""
        return norm.pdf(self.d1()) / (self.S * self.sigma * sqrt(self.T))

    def vega(self):
        """Calculate vega (same for call and put)."""
        return 0.01 * self.S * norm.pdf(self.d1()) * sqrt(self.T)

    def call_theta(self):
        """Calculate call option theta."""
        return 0.01 * (-(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * sqrt(self.T)) 
                       - self.r * self.K * exp(-self.r * self.T) * norm.cdf(self.d2()))

    def put_theta(self):
        """Calculate put option theta."""
        return 0.01 * (-(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * sqrt(self.T)) 
                       + self.r * self.K * exp(-self.r * self.T) * norm.cdf(-self.d2()))

    def call_rho(self):
        """Calculate call option rho."""
        return 0.01 * self.K * self.T * exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_rho(self):
        """Calculate put option rho."""
        return -0.01 * self.K * self.T * exp(-self.r * self.T) * norm.cdf(-self.d2())

    def implied_volatility(self, option_type, market_price, tol=1e-5, max_iterations=100):
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        :param option_type: 'C' for call, 'P' for put
        :param market_price: Observed market price of the option
        :param tol: Tolerance for convergence
        :param max_iterations: Maximum number of iterations
        :return: Implied volatility
        """
        def objective(sigma):
            self.sigma = sigma
            if option_type == 'C':
                return self.call_price() - market_price
            else:
                return self.put_price() - market_price

        return optimize.newton(objective, x0=0.2, tol=tol, maxiter=max_iterations)

def get_float_input(prompt, min_value=0, max_value=None):
    """Get float input from user with validation."""
    while True:
        try:
            value = float(input(prompt))
            if value <= min_value or (max_value is not None and value > max_value):
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a number greater than {min_value}" + 
                  (f" and less than or equal to {max_value}" if max_value is not None else ""))

def get_date_input(prompt):
    """Get date input from user with validation."""
    while True:
        date_str = input(prompt)
        try:
            expiration_date = datetime.strptime(date_str, "%m-%d-%Y")
            current_date = datetime.now()
            if expiration_date <= current_date:
                print("Error: Expiration date must be in the future.")
                continue
            return expiration_date
        except ValueError:
            print("Invalid date format. Please use mm-dd-yyyy.")

def main():
    # Get user inputs
    S = get_float_input("Enter the current stock price: ")
    K = get_float_input("Enter the strike price: ")
    expiration_date = get_date_input("Enter the expiration date (mm-dd-yyyy): ")
    T = (expiration_date - datetime.now()).days / DAYS_PER_YEAR
    r = get_float_input("Enter the risk-free interest rate (as a decimal): ", max_value=1)
    sigma = get_float_input("Enter the volatility (as a decimal): ", max_value=1)

    # Create Black-Scholes Model instance
    bsm = BlackScholesModel(S, K, T, r, sigma)

    # Display input parameters
    params = pd.DataFrame({
        'Parameter': ['Stock Price', 'Strike Price', 'Time to Maturity (years)', 'Risk-free Rate', 'Volatility'],
        'Value': [S, K, T, r, sigma]
    })
    print("\nInput Parameters:")
    print(params.to_string(index=False))

    # Calculate and display option prices
    call_price = bsm.call_price()
    put_price = bsm.put_price()
    print(f"\nCall Option Price: {call_price:.4f}")
    print(f"Put Option Price: {put_price:.4f}")

    # Calculate and display Greeks
    print("\nGreeks:")
    print(f"Call Delta: {bsm.call_delta():.4f}")
    print(f"Put Delta: {bsm.put_delta():.4f}")
    print(f"Gamma: {bsm.gamma():.4f}")
    print(f"Vega: {bsm.vega():.4f}")
    print(f"Call Theta: {bsm.call_theta():.4f}")
    print(f"Put Theta: {bsm.put_theta():.4f}")
    print(f"Call Rho: {bsm.call_rho():.4f}")
    print(f"Put Rho: {bsm.put_rho():.4f}")

    # Calculate implied volatility
    option_type = input("Enter option type for implied volatility calculation (C/P): ").upper()
    while option_type not in ['C', 'P']:
        option_type = input("Invalid input. Enter C for Call or P for Put: ").upper()
    
    market_price = get_float_input("Enter the market price of the option: ")
    try:
        implied_vol = bsm.implied_volatility(option_type, market_price)
        print(f"Implied Volatility: {implied_vol:.4%}")
    except RuntimeError:
        print("Could not calculate implied volatility. The algorithm did not converge.")

    # Plot implied volatility smile
    K_range = np.linspace(S * 0.5, S * 1.5, 50)
    implied_vols = []
    for K in K_range:
        bsm.K = K
        try:
            implied_vol = bsm.implied_volatility(option_type, market_price)
            implied_vols.append(implied_vol)
        except RuntimeError:
            implied_vols.append(np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, implied_vols, marker='o')
    plt.title('Implied Volatility Smile')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
