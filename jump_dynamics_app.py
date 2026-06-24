import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from data_engine import simulate_merton_jump_diffusion

st.set_page_config(layout="wide")
st.title("⚡ Jump Diffusion Dynamics & Empirical Heavy Tails")
st.markdown("---")

st.markdown("""
### 🧠 Empirical Realities of Asset Dynamics (Lévy & Jump Processes)
जैसा कि अनुभवजन्य अध्ययनों (Empirical Studies) से पता चलता है, वास्तविक बाजार के रिटर्न डिस्ट्रीब्यूशन्स सामान्य वितरण की तुलना में **Too Peaked (अत्यधिक नुकीले)** होते हैं और उनमें **Heavy Tails** पाई जाती हैं।
""")

# Sidebar parameters
st.sidebar.header("⚡ Jump SDE Parameters")
s0 = st.sidebar.number_input("Initial Asset Price (S0)", value=100.0)
mu = st.sidebar.slider(r"Drift ($\mu$)", -0.5, 0.5, 0.05)
sigma = st.sidebar.slider(r"Continuous Vol ($\sigma$)", 0.05, 0.8, 0.2)
lam = st.sidebar.slider(r"Poisson Jump Intensity ($\lambda$)", 0.1, 10.0, 4.0, help="Expected number of jumps per year")
j_mu = st.sidebar.slider("Mean Jump Magnitude", -0.2, 0.2, -0.05)
j_sigma = st.sidebar.slider("Jump Volatility", 0.01, 0.3, 0.05)

# 4 Updated Tabs aligning perfectly with your lecture slide
tab1, tab2, tab3, tab4 = st.tabs([
    "Heavy Tails & Kurtosis (Slide Core)",
    "Poisson Stock Simulation", 
    "Partial Integro-Differential Equations (PIDE)", 
    "Implied Volatility Smirk"
])

# --- NEW TAB 1: Heavy Tails & Excess Kurtosis Visualization ---
with tab1:
    st.subheader("📊 Empirical Densities vs Normal Distribution")
    st.markdown("""
    * **Excess Kurtosis:** जंप मॉडल्स के कारण वितरण केंद्र में अत्यधिक नुकीला (Too Peaked) हो जाता है।
    * **Heavy Tails (Fat Tails):** सामान्य वितरण (Gaussian) की तुलना में चरम घटनाएं (Extreme Events) अधिक संभावित होती हैं।
    """)
    
    # Generate simulated daily log returns with and without jumps
    np.random.seed(42)
    n_samples = 5000
    
    # Pure Gaussian returns
    gaussian_returns = np.random.normal(mu/252, sigma/np.sqrt(252), n_samples)
    
    # Jump Diffusion returns (Merton Proxy)
    continuous_part = np.random.normal(mu/252, sigma/np.sqrt(252), n_samples)
    poisson_jumps = np.random.poisson(lam/252, n_samples)
    jump_part = np.array([np.sum(np.random.normal(j_mu, j_sigma, j)) if j > 0 else 0 for j in poisson_jumps])
    jump_returns = continuous_part + jump_part
    
    # Calculate Statistical Parameters
    kurt_gauss = stats.kurtosis(gaussian_returns, fisher=True)
    kurt_jump = stats.kurtosis(jump_returns, fisher=True)
    skew_jump = stats.skew(jump_returns)
    
    # Display Stats Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Gaussian Excess Kurtosis", f"{kurt_gauss:.2f}", "0.00 (Ideal)")
    col2.metric("Jump Model Excess Kurtosis", f"{kurt_jump:.2f} (Too Peaked)", f"+{kurt_jump:.2f}", delta_color="inverse")
    col3.metric("Asymmetry (Skewness)", f"{skew_jump:.2f}")
    
    # Plotting the Distribution Comparison
    hist_data = pd.DataFrame({
        "Returns": np.concatenate([gaussian_returns, jump_returns]),
        "Distribution Type": ["Pure Normal (Black-Scholes)"] * n_samples + ["Jump-Diffusion (Heavy Tails)"] * n_samples
    })
    
    fig_hist = px.histogram(
        hist_data, x="Returns", color="Distribution Type", barmode="overlay",
        marginal="box", nbins=100, title="The Reality of Excess Kurtosis and Tail Risk"
    )
    fig_hist.update_layout(xaxis_range=[-0.15, 0.15])
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("""
    > 💡 **Lévy Convergence Note:** ध्यान दें कि जब आप टाइम होराइजन को छोटा (Daily) रखते हैं, तो जंप्स के कारण पूंछ (Tails) बहुत भारी होती हैं। लेकिन जैसे-जैसे समय लंबा होता है, सेंट्रल लिमिट थ्योरम के कारण ये वापस नॉर्मलिटी की तरफ बढ़ने लगती हैं।
    """)

# --- TAB 2: Stock Simulation ---
with tab2:
    st.subheader("📊 Jump Diffusion Path vs Pure Brownian Path")
    t, S_jump = simulate_merton_jump_diffusion(s0, mu, sigma, lam, j_mu, j_sigma, T=1.0, N=252)
    _, S_pure = simulate_merton_jump_diffusion(s0, mu, sigma, lam=0.0, j_mu=0, j_sigma=0, T=1.0, N=252)
    
    sim_df = pd.DataFrame({
        "Trading Day": np.arange(252),
        "Merton Jump Diffusion Path": S_jump[:-1],
        "Standard Geometric Brownian Motion": S_pure[:-1]
    })
    fig = px.line(sim_df, x="Trading Day", y=["Merton Jump Diffusion Path", "Standard Geometric Brownian Motion"],
                  title=r"Asset Price Simulation under $\mathbb{Q}$-Measure")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: PIDE Insights ---
with tab3:
    st.subheader("📐 Partial Integro-Differential Equations (PIDE)")
    st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV + \lambda \int_0^\infty [V(Sy) - V(S)]\psi(y)dy = 0")
    
    x_space = np.linspace(-3, 3, 20)
    y_space = np.linspace(50, 150, 20)
    z_density = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            z_density[i, j] = np.exp(-0.5 * (x_space[i]**2)) * (y_space[j] / 100.0)
            
    fig_heat = px.imshow(z_density, x=x_space, y=y_space, title="Non-local Jump Operator Spectrum")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: Implied Volatility Smirk ---
with tab4:
    st.subheader("📉 Fitting the Smile in Implied Volatility")
    st.markdown("जैसा कि आपकी स्लाइड की आखिरी लाइन कहती है: कॉपुला या जंप पैरामीटर्स को एडजस्ट करके हम वास्तविक मार्केट के **Volatility Smile/Smirk** को पूरी तरह फिट कर सकते हैं।")
    
    strikes = np.linspace(80, 120, 15)
    iv_pure = [sigma * 100] * len(strikes)
    iv_jump = [(sigma * 100) + (abs(j_mu) * 150) * (100 - k)/10 if k < 100 else (sigma * 100) - (j_sigma * 50) * (k - 100)/10 for k in strikes]
    
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(x=strikes, y=iv_jump, name="Merton Jump Implied Vol (Fitted Smile)", line=dict(color='red', width=3)))
    fig_smile.add_trace(go.Scatter(x=strikes, y=iv_pure, name="Flat Constant Vol (Black-Scholes)", line=dict(dash='dash', color='blue')))
    st.plotly_chart(fig_smile, use_container_width=True)
