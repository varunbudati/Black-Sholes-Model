# Black-Scholes Option Pricing Model
![chrome_0689IPcQEf](https://github.com/user-attachments/assets/37bd7c0b-dfcd-420b-aa2d-591b1fc38cdb)

## Description
This project implements the Black-Scholes option pricing model in Python. It provides a comprehensive tool for calculating option prices, Greeks, and implied volatility for European-style options.

## Features
- Calculate Call and Put option prices
- Compute Greeks (Delta, Gamma, Vega, Theta, Rho)
- Visualize option prices against stock price
- Perform sensitivity analysis
- Display probability distribution of stock price at maturity
- Interactive UI with adjustable parameters
- Dark mode interface

## Requirements
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy
- pip

## Usage
[Click Here](https://varunbudati-black-sholes-model-black-scholes-streamlit-qc7aie.streamlit.app/)

## How to Use
1. Adjust the input parameters:
   - Stock Price
   - Strike Price
   - Time to Maturity (in days)
   - Risk-free Rate (%)
   - Volatility (%)

2. The app will automatically calculate and display:
   - Option Prices (Call and Put)
   - Greeks (Delta, Gamma, Vega, Theta, Rho)

3. Explore different tabs for:
   - Model Visualization
   - Sensitivity Analysis
   - Probability Distribution

## Theory
The Black-Scholes model, developed by Fischer Black and Myron Scholes in 1973, is a mathematical model for pricing European-style options. The model assumes that heavily traded asset prices follow a geometric Brownian motion with constant drift and volatility.

Key assumptions of the model:
- The option is European and can only be exercised at expiration
- No dividends are paid out during the option's life
- Markets are efficient (i.e., market movements cannot be predicted)
- There are no transaction costs in buying the option
- The risk-free rate and volatility of the underlying are known and constant
- The returns on the underlying asset are normally distributed

## Contributing
Contributions to improve the calculator are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request


## Images


![image](https://github.com/user-attachments/assets/667b7d08-3021-4ae4-bf83-6521da13167a)


![image](https://github.com/user-attachments/assets/914d4c34-c48d-4a39-850d-bd0cebe4e00d)
