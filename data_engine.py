"""
Data Engine for Merton Jump-Diffusion Model Simulation
Implements the SDE: dS(t) = μS(t)dt + σS(t)dW(t) + S(t)(J-1)dN(t)
"""

import numpy as np


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
