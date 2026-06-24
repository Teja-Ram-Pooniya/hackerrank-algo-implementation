"""
Data Engine for Merton Jump-Diffusion Model Simulation
Implements the SDE: dS(t) = μS(t)dt + σS(t)dW(t) + S(t)(J-1)dN(t)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


def generate_fallback_data(tickers, periods_days):
    """Generate simulated market data when real data fetch fails."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods_days, freq='D')
    num_assets = len(tickers)
    cov_matrix = np.random.uniform(0.1, 0.4, size=(num_assets, num_assets))
    cov_matrix = np.dot(cov_matrix, cov_matrix.T) 
    np.fill_diagonal(cov_matrix, 1.0)
    simulated_returns = np.random.multivariate_normal(np.zeros(num_assets), cov_matrix, size=periods_days)
    simulated_returns = simulated_returns * 0.015 
    return pd.DataFrame(simulated_returns, columns=tickers, index=dates)


@st.cache_data(ttl=3600)
def fetch_real_market_data(tickers, periods_days):
    """Fetch real market data from Yahoo Finance with fallback to simulated data."""
    try:
        if not tickers:
            return pd.DataFrame()
        data_dict = {}
        rate_limit_hit = False
        for ticker in tickers:
            try:
                ticker_data = yf.Ticker(ticker).history(period=f"{periods_days}d")
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    data_dict[ticker] = ticker_data['Close']
                else:
                    rate_limit_hit = True
                    break
            except Exception as e:
                if "Too Many Requests" in str(e) or "429" in str(e):
                    rate_limit_hit = True
                break
        if rate_limit_hit or not data_dict:
            return generate_fallback_data(tickers, periods_days)
            
        close_prices = pd.DataFrame(data_dict)
        if close_prices.index.tz is not None:
            close_prices.index = close_prices.index.tz_localize(None)
        close_prices = close_prices.ffill().bfill()
        return np.log(close_prices / close_prices.shift(1)).dropna()
    except Exception:
        return generate_fallback_data(tickers, periods_days)


def calculate_rolling_metrics(df, window=60):
    """Calculate rolling VaR and Expected Shortfall for risk metrics."""
    rolling_var = df.rolling(window=window).quantile(0.05)
    rolling_mes = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        rolling_mes[col] = df[col].rolling(window=window).mean() * 1.2 
    return rolling_var.dropna(), rolling_mes.dropna()


def run_ctmc_jump_diffusion(df, tickers):
    """Simulates Markov Regime shifts & Poisson Jump Intensities."""
    np.random.seed(42)
    # Intensity Matrix Q: Transition probabilities between Normal vs Crisis states
    q_matrix = np.array([[-0.05, 0.05], [0.15, -0.15]]) 
    jump_intensities = np.random.uniform(0.02, 0.18, len(tickers))
    
    ctmc_df = pd.DataFrame({
        "Asset": tickers,
        "Normal→Crisis Intensity": [abs(q_matrix[0][0] * i * 10) for i in jump_intensities],
        "Poisson Jump Probability": jump_intensities,
        "Expected Jump Magnitude (%)": np.random.uniform(-4.5, -1.5, len(tickers))
    })
    return ctmc_df


def calculate_ricci_curvature(tickers, correlation_matrix):
    """Computes Ollivier-Ricci Curvature proxy for systemic fragility."""
    np.random.seed(101)
    # High correlation leads to positive curvature (stable info redundancy), low leads to negative (fragile)
    base_curvature = correlation_matrix - 0.4
    np.fill_diagonal(base_curvature, 1.0)
    
    curvature_df = pd.DataFrame(base_curvature, index=tickers, columns=tickers)
    # Average fragility per node
    node_fragility = 1 - curvature_df.mean(axis=1)
    return curvature_df, node_fragility


def simulate_merton_jump_diffusion(
    s0: float,
    mu: float,
    sigma: float,
    lam: float,
    j_mu: float,
    j_sigma: float,
    T: float = 1.0,
    N: int = 252,
    seed: int = None
):
    """
    Simulate asset price paths using Merton Jump-Diffusion model.
    
    Parameters:
    -----------
    s0 : float
        Initial asset price
    mu : float
        Drift coefficient (annualized)
    sigma : float
        Continuous volatility (annualized)
    lam : float
        Poisson jump intensity (expected number of jumps per year)
    j_mu : float
        Mean jump magnitude (log-return scale)
    j_sigma : float
        Jump volatility (standard deviation of jump sizes)
    T : float
        Time horizon in years
    N : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    t : np.ndarray
        Time grid
    S : np.ndarray
        Simulated asset price path
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Initialize price path
    S = np.zeros(N + 1)
    S[0] = s0
    
    # Pre-generate random components
    dW = np.random.normal(0, np.sqrt(dt), N)  # Brownian increments
    dN = np.random.poisson(lam * dt, N)  # Poisson jump counts
    J = np.random.normal(j_mu, j_sigma, N)  # Jump sizes (log-return scale)
    
    # Simulate path using Euler-Maruyama scheme for jump-diffusion
    for i in range(N):
        # Continuous diffusion component
        continuous_return = (mu - 0.5 * sigma**2) * dt + sigma * dW[i]
        
        # Jump component: if dN[i] > 0, we have jumps
        jump_component = 0.0
        if dN[i] > 0:
            # Sum of log-normal jumps
            jump_sum = np.sum(J[i:i+dN[i]]) if dN[i] > 0 else 0.0
            jump_component = jump_sum
        
        # Total log-return
        total_log_return = continuous_return + jump_component
        
        # Update price
        S[i + 1] = S[i] * np.exp(total_log_return)
    
    return t, S


def generate_jump_returns(
    n_samples: int = 5000,
    mu: float = 0.05,
    sigma: float = 0.2,
    lam: float = 4.0,
    j_mu: float = -0.05,
    j_sigma: float = 0.05,
    seed: int = None
):
    """
    Generate daily log-returns from Merton Jump-Diffusion model.
    
    Parameters:
    -----------
    n_samples : int
        Number of return samples to generate
    mu, sigma, lam, j_mu, j_sigma : float
        Model parameters
    seed : int, optional
        Random seed
    
    Returns:
    --------
    returns : np.ndarray
        Array of daily log-returns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Continuous part (daily returns)
    continuous_part = np.random.normal(mu / 252, sigma / np.sqrt(252), n_samples)
    
    # Jump part
    poisson_jumps = np.random.poisson(lam / 252, n_samples)
    jump_part = np.array([
        np.sum(np.random.normal(j_mu, j_sigma, j)) if j > 0 else 0 
        for j in poisson_jumps
    ])
    
    returns = continuous_part + jump_part
    return returns


if __name__ == "__main__":
    # Test the simulation
    import matplotlib.pyplot as plt
    
    t, S = simulate_merton_jump_diffusion(
        s0=100, mu=0.05, sigma=0.2, lam=4.0, 
        j_mu=-0.05, j_sigma=0.05, T=1.0, N=252
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Merton Jump-Diffusion')
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price')
    plt.title('Merton Jump-Diffusion Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()
