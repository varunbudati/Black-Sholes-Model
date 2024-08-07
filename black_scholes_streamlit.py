import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import optimize
import plotly.graph_objs as go

# Constants
DAYS_PER_YEAR = 365.0

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.validate_inputs(S, K, T, r, sigma)
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    @staticmethod
    def validate_inputs(S, K, T, r, sigma):
        if S <= 0 or K <= 0:
            raise ValueError("Stock price and strike price must be positive.")
        if T <= 0:
            raise ValueError("Time to maturity must be positive.")
        if r < 0 or r > 1:
            raise ValueError("Risk-free rate should be between 0 and 1.")
        if sigma <= 0 or sigma > 1:
            raise ValueError("Volatility should be between 0 and 1.")

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) - self.S + self.call_price()

    def call_delta(self):
        return norm.cdf(self.d1())

    def put_delta(self):
        return -norm.cdf(-self.d1())

    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return 0.01 * self.S * norm.pdf(self.d1()) * np.sqrt(self.T)

    def call_theta(self):
        return 0.01 * (-(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T)) 
                       - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))

    def put_theta(self):
        return 0.01 * (-(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T)) 
                       + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))

    def call_rho(self):
        return 0.01 * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_rho(self):
        return -0.01 * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())

    def implied_volatility(self, option_type, market_price, tol=1e-5, max_iterations=100):
        def objective(sigma):
            self.sigma = sigma
            if option_type == 'C':
                return self.call_price() - market_price
            else:
                return self.put_price() - market_price

        return optimize.newton(objective, x0=0.2, tol=tol, maxiter=max_iterations)


# Streamlit app
def main():
    st.set_page_config(page_title="Black-Scholes Option's Pricing Calculator", layout="wide")
    
    st.title("Black-Scholes Option's Pricing Calculator")

    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        stock_price = st.number_input("Stock Price", value=100.0, min_value=0.01, step=0.01)
        strike_price = st.number_input("Strike Price", value=100.0, min_value=0.01, step=0.01)
        maturity_time = st.number_input("Time to Maturity (days)", value=30, min_value=1, step=1)
    
    with col2:
        risk_free_rate = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=0.0, max_value=100.0, step=0.1)

    # Convert inputs
    maturity_time = maturity_time / DAYS_PER_YEAR
    risk_free_rate = risk_free_rate / 100
    volatility = volatility / 100

    # Create BSM object
    bsm = BlackScholesModel(stock_price, strike_price, maturity_time, risk_free_rate, volatility)

    # Calculate option prices
    call_price = bsm.call_price()
    put_price = bsm.put_price()

    # Display option prices
    st.subheader("Option Prices")
    col1, col2 = st.columns(2)
    col1.metric("Call Option Price", f"${call_price:.2f}")
    col2.metric("Put Option Price", f"${put_price:.2f}")

    # Calculate and display Greeks
    st.subheader("Greeks")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Call Delta", f"{bsm.call_delta():.4f}")
    col1.metric("Put Delta", f"{bsm.put_delta():.4f}")
    col2.metric("Gamma", f"{bsm.gamma():.4f}")
    col2.metric("Vega", f"{bsm.vega():.4f}")
    col3.metric("Call Theta", f"{bsm.call_theta():.4f}")
    col3.metric("Put Theta", f"{bsm.put_theta():.4f}")
    col4.metric("Call Rho", f"{bsm.call_rho():.4f}")
    col4.metric("Put Rho", f"{bsm.put_rho():.4f}")

    # Visualizations
    st.subheader("Black-Scholes Model Visualization")
    stock_prices = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)
    call_prices = [BlackScholesModel(s, strike_price, maturity_time, risk_free_rate, volatility).call_price() for s in stock_prices]
    put_prices = [BlackScholesModel(s, strike_price, maturity_time, risk_free_rate, volatility).put_price() for s in stock_prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=call_prices, mode='lines', name='Call Option'))
    fig.add_trace(go.Scatter(x=stock_prices, y=put_prices, mode='lines', name='Put Option'))
    fig.update_layout(title='Option Prices vs Stock Price', xaxis_title='Stock Price', yaxis_title='Option Price')
    st.plotly_chart(fig)

    # Sensitivity Analysis
    st.subheader("Sensitivity Analysis")
    volatilities = np.linspace(max(0.01, volatility * 0.5), min(1, volatility * 1.5), 50)
    call_prices_vol = [BlackScholesModel(stock_price, strike_price, maturity_time, risk_free_rate, v).call_price() for v in volatilities]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=call_prices, mode='lines', name='Call Price vs Stock Price'))
    fig.add_trace(go.Scatter(x=volatilities, y=call_prices_vol, mode='lines', name='Call Price vs Volatility'))
    fig.update_layout(title='Sensitivity Analysis', xaxis_title='Parameter Value', yaxis_title='Call Option Price')
    st.plotly_chart(fig)

    # Probability Distribution
    st.subheader("Probability Distribution of Stock Price at Maturity")
    expected_price = stock_price * np.exp(risk_free_rate * maturity_time)
    prices = np.linspace(expected_price * 0.5, expected_price * 1.5, 100)
    pdf = norm.pdf(prices, loc=expected_price, scale=volatility * stock_price * np.sqrt(maturity_time))
    cdf = norm.cdf(prices, loc=expected_price, scale=volatility * stock_price * np.sqrt(maturity_time))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pdf, mode='lines', name='PDF'))
    fig.add_trace(go.Scatter(x=prices, y=cdf, mode='lines', name='CDF'))
    fig.update_layout(title='Probability Distribution of Stock Price at Maturity', xaxis_title='Stock Price', yaxis_title='Probability')
    st.plotly_chart(fig)

    # Implied Volatility
    st.subheader("Implied Volatility Calculator")
    col1, col2, col3 = st.columns(3)
    with col1:
        market_price = st.number_input("Market Price", value=10.0, min_value=0.01, step=0.01)
    with col2:
        option_type = st.selectbox("Option Type", ['Call', 'Put'])
    with col3:
        if st.button("Calculate Implied Volatility"):
            try:
                implied_vol = bsm.implied_volatility('C' if option_type == 'Call' else 'P', market_price)
                st.success(f"Implied Volatility: {implied_vol:.2%}")
            except:
                st.error("Could not calculate implied volatility. Please check your inputs.")

if __name__ == "__main__":
    main()