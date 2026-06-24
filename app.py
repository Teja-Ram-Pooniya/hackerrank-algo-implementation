# -*- coding: utf-8 -*-
"""
Main Entry Point for Quant Network & Systemic Risk Analytics
Integrates: Data Loading, Backtesting, Jump Dynamics, and Network Analysis
"""
import numpy as np
import pandas as pd
import streamlit as st
from data_engine import (
    fetch_real_market_data, 
    calculate_rolling_metrics, 
    calculate_ricci_curvature, 
    run_ctmc_jump_diffusion,
    simulate_merton_jump_diffusion
)
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Quant Network & Systemic Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Configuration ---
st.sidebar.title("🌍 Global Market Config")
ticker_options = {
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "Apple (AAPL)": "AAPL",
    "NVIDIA (NVDA)": "NVDA",
    "Gold (GLD)": "GLD"
}

selected_labels = st.sidebar.multiselect(
    "Select Assets for Analysis",
    options=list(ticker_options.keys()),
    default=["Apple (AAPL)", "NVIDIA (NVDA)"]
)

tickers = [ticker_options[label] for label in selected_labels]
lookback_period = st.sidebar.slider("Historical Lookback (Days)", 200, 1500, 750)

# --- Main App Logic ---
st.title("📊 Quant Network & Systemic Risk Analytics")
st.markdown("---")

if len(tickers) >= 2:
    with st.spinner("Fetching live market data via yfinance..."):
        df_returns = fetch_real_market_data(tickers, lookback_period)
        
    if not df_returns.empty:
        # Store in Session State for other modules to access if needed
        st.session_state['market_data'] = df_returns
        st.session_state['tickers'] = tickers
        
        st.success(f"✅ Successfully loaded {df_returns.shape[0]} trading days for {len(tickers)} assets.")
        
        # Display Initial Volatility Metrics
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            with cols[i]:
                vol = df_returns[t].std() * np.sqrt(252) * 100
                st.metric(label=f"{t.split('(')[0].strip()} Ann. Vol", value=f"{vol:.2f}%")
        
        st.markdown("---")
        
        # --- Navigation Tabs ---
        tab_network, tab_backtest, tab_dynamics = st.tabs([
            "🕸️ Network & Systemic Risk", 
            "⚡ Jump-Signal Backtest", 
            "🧠 Jump Diffusion Dynamics"
        ])
        
        # TAB 1: Network & Systemic Risk (Original App Logic)
        with tab_network:
            st.subheader("🕸️ Correlation & Systemic Fragility")
            
            # Correlation Matrix
            corr_matrix = df_returns.corr()
            st.write("#### Asset Correlation Matrix")
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))
            
            # Ricci Curvature
            curvature_df, node_fragility = calculate_ricci_curvature(tickers, corr_matrix.values)
            st.write("#### 📐 Ollivier-Ricci Curvature (Systemic Fragility Proxy)")
            st.dataframe(node_fragility.to_frame(name="Fragility Score"))
            
            # CTMC Jump Diffusion Stats
            st.write("#### 🎲 CTMC Regime Shift & Jump Intensities")
            ctmc_data = run_ctmc_jump_diffusion(df_returns, tickers)
            st.dataframe(ctmc_data)

        # TAB 2: Backtesting Engine
        with tab_backtest:
            st.subheader("⚡ SDE-Driven Signal Generation")
            st.markdown("""
            यह इंजन आपकी जंप-डिफ्यूजन SDE स्लाइस के आधार पर काम करता है। जब दैनिक लॉग-रिटर्न $X(t)$ एक निश्चित थ्रेशोल्ड को पार करता है, तो मॉडल उसे एक **Poisson Jump** मानकर काउंटर-ट्रेंड सिग्नल जेनरेट करता है।
            """)
            
            # Backtest Sidebar Settings
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Backtest Parameters")
            target_asset = st.sidebar.selectbox("Select Asset to Trade", tickers, key="backtest_asset")
            jump_threshold = st.sidebar.slider("Jump Detection Threshold (Z-Score)", 1.5, 4.0, 2.5, key="jump_thresh")
            allocation = st.sidebar.slider("Portfolio Allocation per Trade (%)", 10, 100, 50, key="alloc") / 100.0
            
            # Engine Logic
            returns = df_returns[target_asset].copy()
            rolling_std = returns.rolling(window=20).std()
            rolling_mean = returns.rolling(window=20).mean()
            z_scores = (returns - rolling_mean) / rolling_std
            
            signals = np.zeros(len(returns))
            signals[z_scores < -jump_threshold] = 1   # Long
            signals[z_scores > jump_threshold] = -1   # Short
            
            strategy_returns = signals.astype(float) * returns.shift(-1) * allocation
            strategy_returns = strategy_returns.fillna(0)
            
            cum_market = (1 + returns).cumprod() * 10000
            cum_strategy = (1 + strategy_returns).cumprod() * 10000
            
            # Metrics
            total_trades = np.count_nonzero(signals)
            sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252) if total_trades > 0 else 0
            max_dd = (((cum_strategy.cummax() - cum_strategy) / cum_strategy.cummax()).max()) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Jumps Detected", f"{total_trades}")
            m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            m3.metric("Max Drawdown", f"{max_dd:.2f}%")
            m4.metric("End Value", f"${cum_strategy.iloc[-1]:.2f}")
            
            # Plot Equity Curve
            perf_df = pd.DataFrame({
                "Date": returns.index,
                "Strategy": cum_strategy.values,
                "Benchmark": cum_market.values
            }).set_index("Date")
            st.plotly_chart(px.line(perf_df, title=f"Backtest: {target_asset}", labels={"value": "Portfolio Value ($)", "index": "Date"}), use_container_width=True)
            
            # Jump Isolation Plot
            st.write("#### ⚡ Time-Series Jump Isolation Plot")
            fig_jumps = go.Figure()
            fig_jumps.add_trace(go.Scatter(
                x=returns.index, 
                y=returns.values, 
                name="Daily Log Returns X(t)", 
                line=dict(color='rgba(128, 128, 128, 0.5)')
            ))
            
            jump_dates = returns.index[signals != 0]
            jump_values = returns[signals != 0]
            fig_jumps.add_trace(go.Scatter(
                x=jump_dates, 
                y=jump_values, 
                mode='markers', 
                name="Detected Poisson Jumps", 
                marker=dict(color='red', size=8)
            ))
            fig_jumps.update_layout(title="Isolated Jump Components from Brownian Motion Noise", xaxis_title="Date", yaxis_title="Log Return")
            st.plotly_chart(fig_jumps, use_container_width=True)

        # TAB 3: Jump Dynamics
        with tab_dynamics:
            st.subheader("🧠 Empirical Realities of Asset Dynamics")
            st.markdown("""
            जैसा कि अनुभवजन्य अध्ययनों से पता चलता है, वास्तविक बाजार के रिटर्न डिस्ट्रीब्यूशन्स सामान्य वितरण की तुलना में **Too Peaked** होते हैं और उनमें **Heavy Tails** पाई जाती हैं।
            """)
            
            st.sidebar.header("⚡ SDE Parameters")
            s0 = st.sidebar.number_input("Initial Price (S0)", value=100.0, key="s0")
            mu = st.sidebar.slider("Drift (μ)", -0.5, 0.5, 0.05, key="mu")
            sigma = st.sidebar.slider("Vol (σ)", 0.05, 0.8, 0.2, key="sigma")
            lam = st.sidebar.slider("Jump Intensity (λ)", 0.1, 10.0, 4.0, key="lam")
            j_mu = st.sidebar.slider("Mean Jump Magnitude", -0.2, 0.2, -0.05, key="j_mu")
            j_sigma = st.sidebar.slider("Jump Vol", 0.01, 0.3, 0.05, key="j_sig")
            
            # Heavy Tails Demo
            n_samples = 5000
            gaussian_returns = np.random.normal(mu/252, sigma/np.sqrt(252), n_samples)
            continuous_part = np.random.normal(mu/252, sigma/np.sqrt(252), n_samples)
            poisson_jumps = np.random.poisson(lam/252, n_samples)
            jump_part = np.array([np.sum(np.random.normal(j_mu, j_sigma, j)) if j > 0 else 0 for j in poisson_jumps])
            jump_returns = continuous_part + jump_part
            
            kurt_jump = stats.kurtosis(jump_returns, fisher=True)
            skew_jump = stats.skew(jump_returns)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Gaussian Excess Kurtosis", "0.00 (Ideal)")
            col2.metric("Jump Model Excess Kurtosis", f"{kurt_jump:.2f} (Too Peaked)")
            col3.metric("Asymmetry (Skewness)", f"{skew_jump:.2f}")
            
            hist_data = pd.DataFrame({
                "Returns": np.concatenate([gaussian_returns, jump_returns]),
                "Type": ["Pure Normal (Black-Scholes)"] * n_samples + ["Jump-Diffusion (Heavy Tails)"] * n_samples
            })
            fig_hist = px.histogram(
                hist_data, x="Returns", color="Type", barmode="overlay",
                marginal="box", nbins=100, title="The Reality of Excess Kurtosis and Tail Risk"
            )
            fig_hist.update_layout(xaxis_range=[-0.15, 0.15])
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("""
            > 💡 **Lévy Convergence Note:** ध्यान दें कि जब आप टाइम होराइजन को छोटा (Daily) रखते हैं, तो जंप्स के कारण पूंछ (Tails) बहुत भारी होती हैं। लेकिन जैसे-जैसे समय लंबा होता है, सेंट्रल लिमिट थ्योरम के कारण ये वापस नॉर्मलिटी की तरफ बढ़ने लगती हैं।
            """)
            
            # Simulation Path
            t, S_jump = simulate_merton_jump_diffusion(s0, mu, sigma, lam, j_mu, j_sigma, T=1.0, N=252)
            _, S_pure = simulate_merton_jump_diffusion(s0, mu, sigma, lam=0.0, j_mu=0, j_sigma=0, T=1.0, N=252)
            
            sim_df = pd.DataFrame({
                "Trading Day": np.arange(252),
                "Merton Jump Diffusion Path": S_jump,
                "Standard Geometric Brownian Motion": S_pure
            })
            fig_sim = px.line(sim_df, x="Trading Day", y=["Merton Jump Diffusion Path", "Standard Geometric Brownian Motion"],
                              title="Asset Price Simulation under Jump-Diffusion Model")
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # PIDE Section
            st.write("#### 📐 Partial Integro-Differential Equations (PIDE)")
            st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV + \lambda \int_0^\infty [V(Sy) - V(S)]\psi(y)dy = 0")
            
            x_space = np.linspace(-3, 3, 20)
            y_space = np.linspace(50, 150, 20)
            z_density = np.zeros((20, 20))
            for i in range(20):
                for j in range(20):
                    z_density[i, j] = np.exp(-0.5 * (x_space[i]**2)) * (y_space[j] / 100.0)
                    
            fig_heat = px.imshow(z_density, x=x_space, y=y_space, title="Non-local Jump Operator Spectrum")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Implied Volatility Smirk
            st.write("#### 📉 Fitting the Smile in Implied Volatility")
            strikes = np.linspace(80, 120, 15)
            iv_pure = [sigma * 100] * len(strikes)
            iv_jump = [(sigma * 100) + (abs(j_mu) * 150) * (100 - k)/10 if k < 100 else (sigma * 100) - (j_sigma * 50) * (k - 100)/10 for k in strikes]
            
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(x=strikes, y=iv_jump, name="Merton Jump Implied Vol (Fitted Smile)", line=dict(color='red', width=3)))
            fig_smile.add_trace(go.Scatter(x=strikes, y=iv_pure, name="Flat Constant Vol (Black-Scholes)", line=dict(dash='dash', color='blue')))
            st.plotly_chart(fig_smile, use_container_width=True)

    else:
        st.error("No data fetched. Please check your internet connection or tickers.")
else:
    st.warning("⚠️ Please select at least 2 assets to initialize the risk & network pipelines.")
