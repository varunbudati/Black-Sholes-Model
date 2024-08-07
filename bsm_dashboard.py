import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy import optimize
import dash_bootstrap_components as dbc




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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Black-Scholes Option's Pricing Calculator", className="text-center mb-4"), width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Switch(
                    id="theme-switch",
                    label="Dark Mode",
                    value=False,
                    className="mb-3"
                )
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Input Parameters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Stock Price:"),
                                dbc.Input(id='stock-price', type='number', value=100, min=0, step=0.01),
                                dbc.Tooltip("The current price of the underlying stock.", target="stock-price"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Strike Price:"),
                                dbc.Input(id='strike-price', type='number', value=100, min=0, step=0.01),
                                dbc.Tooltip("The price at which the option can be exercised.", target="strike-price"),
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Time to Maturity (days):"),
                                dbc.Input(id='maturity-time', type='number', value=30, min=1, step=1),
                                dbc.Tooltip("The time remaining until the option expires.", target="maturity-time"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Risk-free Rate (%):"),
                                dbc.Input(id='risk-free-rate', type='number', value=5, min=0, max=100, step=0.1),
                                dbc.Tooltip("The theoretical rate of return of an investment with zero risk.", target="risk-free-rate"),
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Volatility (%):"),
                                dbc.Input(id='volatility', type='number', value=20, min=0, max=100, step=0.1),
                                dbc.Tooltip("A measure of the amount of uncertainty or risk about the size of changes in a security's value.", target="volatility"),
                            ], width=6),
                            dbc.Col([
                                dbc.Button("Calculate", id='calculate-button', color="primary", className="mt-4")
                            ], width=6)
                        ], className="mb-3")
                    ])
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("Option Prices"),
                    dbc.CardBody([
                        html.Div(id='option-prices')
                    ])
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("Greeks"),
                    dbc.CardBody([
                        html.Div(id='greeks-output')
                    ])
                ])
            ], md=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Black-Scholes Model Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='bsm-graph'),
                        dbc.Label("Adjust Stock Price:"),
                        dcc.Slider(id='stock-price-slider', min=50, max=150, step=1, value=100, 
                                   marks={i: str(i) for i in range(50, 151, 10)})
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader("Implied Volatility"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Market Price:"),
                                dbc.Input(id='market-price', type='number', value=10, min=0, step=0.01)
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Option Type:"),
                                dcc.Dropdown(id='option-type', options=[
                                    {'label': 'Call', 'value': 'C'},
                                    {'label': 'Put', 'value': 'P'}
                                ], value='C')
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Button("Calculate Implied Volatility", id='calc-iv-button', color="secondary", className="mb-3"),
                        html.Div(id='implied-volatility-output')
                    ])
                ], className="mt-4"),
                dbc.Card([
                    dbc.CardHeader("Sensitivity Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='sensitivity-graph')
                    ])
                ], className="mt-4"),
                dbc.Card([
                    dbc.CardHeader("Probability Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id='pdf-cdf-graph')
                    ])
                ], className="mt-4")
            ], md=8)
        ])
    ], fluid=True, id="main-container")

app.layout = create_layout()

@app.callback(
    Output('sensitivity-graph', 'figure'),
    [Input('calculate-button', 'n_clicks'),
     Input('theme-switch', 'value')],
    [State('stock-price', 'value'),
     State('strike-price', 'value'),
     State('maturity-time', 'value'),
     State('risk-free-rate', 'value'),
     State('volatility', 'value')]
)
def update_sensitivity_graph(n_clicks, dark_mode, stock_price, strike_price, maturity_time, risk_free_rate, volatility):
    maturity_time = maturity_time / DAYS_PER_YEAR
    risk_free_rate = risk_free_rate / 100
    volatility = volatility / 100

    # Generate range of values for each parameter
    stock_prices = np.linspace(stock_price * 0.5, stock_price * 1.5, 50)
    volatilities = np.linspace(max(0.01, volatility * 0.5), min(1, volatility * 1.5), 50)

    # Calculate option prices for different stock prices and volatilities
    call_prices_stock = [BlackScholesModel(s, strike_price, maturity_time, risk_free_rate, volatility).call_price() for s in stock_prices]
    call_prices_vol = [BlackScholesModel(stock_price, strike_price, maturity_time, risk_free_rate, v).call_price() for v in volatilities]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=call_prices_stock, mode='lines', name='Call Price vs Stock Price'))
    fig.add_trace(go.Scatter(x=volatilities, y=call_prices_vol, mode='lines', name='Call Price vs Volatility'))

    fig.update_layout(
        title='Sensitivity Analysis',
        xaxis_title='Parameter Value',
        yaxis_title='Call Option Price',
        legend_title='Parameter',
        hovermode='x unified'
    )

    if dark_mode:
        fig.update_layout(template='plotly_dark')
    else:
        fig.update_layout(template='plotly_white')

    return fig

@app.callback(
    Output('pdf-cdf-graph', 'figure'),
    [Input('calculate-button', 'n_clicks'),
     Input('theme-switch', 'value')],
    [State('stock-price', 'value'),
     State('strike-price', 'value'),
     State('maturity-time', 'value'),
     State('risk-free-rate', 'value'),
     State('volatility', 'value')]
)
def update_pdf_cdf_graph(n_clicks, dark_mode, stock_price, strike_price, maturity_time, risk_free_rate, volatility):
    maturity_time = maturity_time / DAYS_PER_YEAR
    risk_free_rate = risk_free_rate / 100
    volatility = volatility / 100

    # Calculate expected stock price at maturity
    expected_price = stock_price * np.exp(risk_free_rate * maturity_time)

    # Generate range of potential stock prices
    prices = np.linspace(expected_price * 0.5, expected_price * 1.5, 100)

    # Calculate PDF and CDF
    pdf = norm.pdf(prices, loc=expected_price, scale=volatility * stock_price * np.sqrt(maturity_time))
    cdf = norm.cdf(prices, loc=expected_price, scale=volatility * stock_price * np.sqrt(maturity_time))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pdf, mode='lines', name='PDF'))
    fig.add_trace(go.Scatter(x=prices, y=cdf, mode='lines', name='CDF'))

    fig.update_layout(
        title='Probability Distribution of Stock Price at Maturity',
        xaxis_title='Stock Price',
        yaxis_title='Probability',
        legend_title='Distribution',
        hovermode='x unified'
    )

    if dark_mode:
        fig.update_layout(template='plotly_dark')
    else:
        fig.update_layout(template='plotly_white')

    return fig

def toggle_theme(dark_mode):
    if dark_mode:
        return {"background-color": "#222", "color": "white"}
    else:
        return {"background-color": "white", "color": "black"}

@app.callback(
    [Output('bsm-graph', 'figure'),
     Output('greeks-output', 'children'),
     Output('option-prices', 'children')],
    [Input('calculate-button', 'n_clicks'),
     Input('stock-price-slider', 'value'),
     Input('theme-switch', 'value')],  
    [State('stock-price', 'value'),
     State('strike-price', 'value'),
     State('maturity-time', 'value'),
     State('risk-free-rate', 'value'),
     State('volatility', 'value')]
)
def update_dashboard(n_clicks, slider_value, dark_mode, stock_price, strike_price, maturity_time, risk_free_rate, volatility):
    stock_price = slider_value  # Use the slider value for stock price
    
    # Convert inputs to the correct format
    maturity_time = maturity_time / DAYS_PER_YEAR  # Convert days to years
    risk_free_rate = risk_free_rate / 100  # Convert percentage to decimal
    volatility = volatility / 100  # Convert percentage to decimal
    
    bsm = BlackScholesModel(stock_price, strike_price, maturity_time, risk_free_rate, volatility)
    
    # Generate a range of stock prices for the x-axis
    stock_prices = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)
    
    # Calculate call and put prices for each stock price
    call_prices = [BlackScholesModel(s, strike_price, maturity_time, risk_free_rate, volatility).call_price() for s in stock_prices]
    put_prices = [BlackScholesModel(s, strike_price, maturity_time, risk_free_rate, volatility).put_price() for s in stock_prices]
    
    # Create the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=call_prices, mode='lines', name='Call Option'))
    fig.add_trace(go.Scatter(x=stock_prices, y=put_prices, mode='lines', name='Put Option'))
    
    fig.update_layout(
        title='Option Prices vs Stock Price',
        xaxis_title='Stock Price',
        yaxis_title='Option Price',
        legend_title='Option Type',
        hovermode='x unified'
    )
    
    if dark_mode:
        fig.update_layout(
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font_color='white'
        )
    else:
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black'
        )
    
    # Calculate Greeks
    greeks = html.Div([
        dbc.Row([
            dbc.Col([
                html.P(f"Call Delta: {bsm.call_delta():.4f}"),
                html.P(f"Put Delta: {bsm.put_delta():.4f}"),
                html.P(f"Gamma: {bsm.gamma():.4f}"),
                html.P(f"Vega: {bsm.vega():.4f}")
            ], width=6),
            dbc.Col([
                html.P(f"Call Theta: {bsm.call_theta():.4f}"),
                html.P(f"Put Theta: {bsm.put_theta():.4f}"),
                html.P(f"Call Rho: {bsm.call_rho():.4f}"),
                html.P(f"Put Rho: {bsm.put_rho():.4f}")
            ], width=6)
        ])
    ])
    
    # Calculate option prices
    option_prices = html.Div([
        html.P(f"Call Option Price: ${bsm.call_price():.2f}"),
        html.P(f"Put Option Price: ${bsm.put_price():.2f}")
    ])
    
    return fig, greeks, option_prices

@app.callback(
    Output('implied-volatility-output', 'children'),
    [Input('calc-iv-button', 'n_clicks')],
    [State('stock-price', 'value'),
     State('strike-price', 'value'),
     State('maturity-time', 'value'),
     State('risk-free-rate', 'value'),
     State('market-price', 'value'),
     State('option-type', 'value')]
)
def calculate_implied_volatility(n_clicks, stock_price, strike_price, maturity_time, risk_free_rate, market_price, option_type):
    if n_clicks is None:
        return ""
    
    maturity_time = maturity_time / DAYS_PER_YEAR
    risk_free_rate = risk_free_rate / 100
    
    bsm = BlackScholesModel(stock_price, strike_price, maturity_time, risk_free_rate, 0.2)  # Initial guess for volatility
    
    try:
        implied_vol = bsm.implied_volatility(option_type, market_price)
        return f"Implied Volatility: {implied_vol:.2%}"
    except:
        return "Could not calculate implied volatility. Please check your inputs."




if __name__ == '__main__':
    app.run_server(debug=True)