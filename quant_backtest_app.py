import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Quantitative Backtesting & Jump-Signal Engine")
st.markdown("---")

if 'market_data' not in st.session_state:
    st.error("Please load asset data from the Home page first.")
else:
    df_returns = st.session_state['market_data']
    tickers = st.session_state['tickers']
    
    st.markdown(r"""
    ### ⚡ SDE-Driven Signal Generation
    यह इंजन आपकी जंप-डिफ्यूजन SDE स्लाइस के आधार पर काम करता है। जब दैनिक लॉग-रिटर्न $X(t)$ एक निश्चित थ्रेशोल्ड (जैसे $3\sigma$) को पार करता है, तो मॉडल उसे एक **Poisson Jump ($J dX_{\mathcal{P}}$)** मानकर काउंटर-ट्रेंड या हेजिंग सिग्नल जेनरेट करता है।
    """)
    
    # Backtest Sidebar Settings
    st.sidebar.header("⚙️ Strategy Parameters")
    target_asset = st.sidebar.selectbox("Select Asset to Trade", tickers)
    jump_threshold = st.sidebar.slider("Jump Detection Threshold (Z-Score)", 1.5, 4.0, 2.5)
    allocation = st.sidebar.slider("Portfolio Allocation per Trade (%)", 10, 100, 50) / 100.0
    
    # Backtest Engine Logic
    returns = df_returns[target_asset].copy()
    rolling_std = returns.rolling(window=20).std()
    rolling_mean = returns.rolling(window=20).mean()
    
    # Z-Score to isolate Jump Processes
    z_scores = (returns - rolling_mean) / rolling_std
    
    # Signal Logic: Buy when extreme down-jump (Mean reversion asset pricing), Short on extreme up-jump
    signals = np.zeros(len(returns))
    signals[z_scores < -jump_threshold] = 1   # Long Signal
    signals[z_scores > jump_threshold] = -1   # Short Signal
    
    # Strategy Returns (Shift signals by 1 day to avoid lookahead bias)
    strategy_returns = signals.astype(float) * returns.shift(-1) * allocation
    strategy_returns = strategy_returns.fillna(0)
    
    # Cumulative Performance
    cum_market = (1 + returns).cumprod() * 10000
    cum_strategy = (1 + strategy_returns).cumprod() * 10000
    
    # Performance Metrics
    total_trades = np.count_nonzero(signals)
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252) if total_trades > 0 else 0
    max_dd = (((cum_strategy.cummax() - cum_strategy) / cum_strategy.cummax()).max()) * 100
    
    # Display Analytics Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Poisson Jumps Detected", f"{total_trades} Events")
    col2.metric("Strategy Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Max Strategy Drawdown", f"{max_dd:.2f}%")
    col4.metric("Ending Portfolio Value", f"${cum_strategy.iloc[-1]:.2f}", f"Initial: $10,000")
    
    # Visualizing Equity Curves
    st.write("#### 📊 Strategy Performance vs Benchmark Buy & Hold")
    performance_df = pd.DataFrame({
        "Date": returns.index,
        "Jump Signal Strategy (PnL)": cum_strategy.values,
        "Benchmark Buy & Hold": cum_market.values
    }).set_index("Date")
    
    fig_equity = px.line(performance_df, labels={"value": "Portfolio Value ($)", "index": "Date"},
                         title=f"Merton Jump Counter-Trend Backtest: {target_asset}")
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Visualizing the Isolated Poisson Jumps
    st.write("#### ⚡ Time-Series Jump Isolation Plot")
    fig_jumps = go.Figure()
    fig_jumps.add_trace(go.Scatter(
    x=returns.index, 
    y=returns.values, 
    name="Daily Log Returns X(t)", 
    line=dict(color='rgba(128, 128, 128, 0.5)') # rgba() का उपयोग कर ट्रांसपेरेंसी सेट की
    ))
    
    # Highlight isolated jumps
    jump_dates = returns.index[signals != 0]
    jump_values = returns[signals != 0]
    fig_jumps.add_trace(go.Scatter(x=jump_dates, y=jump_values, mode='markers', name=r"Detected Poisson Jumps ($dX_{\mathcal{P}}$)", marker=dict(color='red', size=8)))
    
    fig_jumps.update_layout(title="Isolated Jump Components from Brownian Motion Noise", xaxis_title="Date", yaxis_title="Log Return")
    st.plotly_chart(fig_jumps, use_container_width=True)
