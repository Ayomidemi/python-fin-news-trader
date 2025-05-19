import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

def plot_sentiment_over_time(news_data):
    """
    Plot sentiment analysis results over time
    
    Args:
        news_data (list): List of news article dictionaries with sentiment scores
    """
    if not news_data:
        st.info("No news data available for visualization")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(news_data)
    
    # Ensure we have date and sentiment columns
    if 'date' not in df.columns or 'sentiment' not in df.columns:
        st.warning("News data missing required columns for visualization")
        return
    
    # Group by ticker and date
    if 'ticker' in df.columns:
        # Resample to daily frequency and calculate mean sentiment
        ticker_groups = df.groupby('ticker')
        
        for ticker, group in ticker_groups:
            group = group.sort_values('date')
            
            fig = px.line(
                x=group['date'], 
                y=group['sentiment'],
                labels={'x': 'Date', 'y': 'Sentiment Score'},
                color_discrete_sequence=['#1f77b4'],
            )
            
            fig.update_layout(
                title=f"Sentiment Trend for {ticker}",
                hovermode="x unified",
                height=300,
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
            
            # Set y-axis range
            fig.update_yaxes(range=[-1, 1])
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Overall sentiment
        df = df.sort_values('date')
        
        fig = px.line(
            x=df['date'], 
            y=df['sentiment'],
            labels={'x': 'Date', 'y': 'Sentiment Score'},
            color_discrete_sequence=['#1f77b4'],
        )
        
        fig.update_layout(
            title="Overall Sentiment Trend",
            hovermode="x unified",
            height=300,
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Set y-axis range
        fig.update_yaxes(range=[-1, 1])
        
        st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_performance(performance_data):
    """
    Plot portfolio performance over time
    
    Args:
        performance_data (list): List of portfolio performance snapshots
    """
    if not performance_data:
        st.info("No performance data available for visualization")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_data)
    
    # Extract required columns
    df = df[['timestamp', 'cash', 'positions_value', 'total_value']]
    
    # Create figure
    fig = go.Figure()
    
    # Add total value line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_value'],
        mode='lines',
        name='Total Value',
        line=dict(color='green', width=2)
    ))
    
    # Add cash line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cash'],
        mode='lines',
        name='Cash',
        line=dict(color='blue', width=1.5, dash='dot')
    ))
    
    # Add positions value line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['positions_value'],
        mode='lines',
        name='Positions Value',
        line=dict(color='orange', width=1.5, dash='dot')
    ))
    
    # Add baseline at 100,000
    fig.add_hline(y=100000, line_dash="dash", line_color="gray", opacity=0.7)
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_price_with_signals(ticker, price_data, signals):
    """
    Plot stock price chart with trading signals
    
    Args:
        ticker (str): Stock ticker symbol
        price_data (DataFrame): Stock price data
        signals (list): List of trading signals for this stock
    """
    if price_data.empty:
        st.info(f"No price data available for {ticker}")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name=ticker
    ))
    
    # Add moving averages if available
    if 'MA20' in price_data.columns:
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='blue', width=1)
        ))
    
    if 'MA50' in price_data.columns:
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='orange', width=1)
        ))
    
    # Add signals
    buy_signals = [s for s in signals if s['action'] == 'BUY']
    sell_signals = [s for s in signals if s['action'] == 'SELL']
    
    if buy_signals:
        buy_dates = [s['timestamp'] for s in buy_signals]
        buy_prices = [s['price'] for s in buy_signals]
        
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='Buy Signal',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up',
                line=dict(width=2, color='darkgreen')
            )
        ))
    
    if sell_signals:
        sell_dates = [s['timestamp'] for s in sell_signals]
        sell_prices = [s['price'] for s in sell_signals]
        
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='Sell Signal',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down',
                line=dict(width=2, color='darkred')
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Chart with Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_distribution(news_data):
    """
    Plot distribution of sentiment scores
    
    Args:
        news_data (list): List of news article dictionaries with sentiment scores
    """
    if not news_data:
        st.info("No news data available for visualization")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(news_data)
    
    # Ensure we have sentiment column
    if 'sentiment' not in df.columns:
        st.warning("News data missing sentiment column for visualization")
        return
    
    # Create histogram
    fig = px.histogram(
        df,
        x='sentiment',
        nbins=20,
        color_discrete_sequence=['skyblue'],
        labels={'sentiment': 'Sentiment Score'}
    )
    
    fig.update_layout(
        title="Distribution of News Sentiment Scores",
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        xaxis=dict(range=[-1, 1]),
        bargap=0.1
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_entity_frequency(news_data, entity_type='ORG'):
    """
    Plot frequency of named entities in news data
    
    Args:
        news_data (list): List of news article dictionaries
        entity_type (str): Type of entity to visualize (ORG, PERSON, GPE, etc.)
    """
    if not news_data:
        st.info("No news data available for entity visualization")
        return
    
    # Extract entities
    entity_counts = {}
    
    for article in news_data:
        if 'entities' in article and entity_type in article['entities']:
            for entity in article['entities'][entity_type]:
                name = entity['text']
                count = entity['count']
                
                if name in entity_counts:
                    entity_counts[name] += count
                else:
                    entity_counts[name] = count
    
    if not entity_counts:
        st.info(f"No {entity_type} entities found in the news data")
        return
    
    # Sort by frequency
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    top_entities = sorted_entities[:10]  # Show top 10
    
    # Create bar chart
    entity_names = [e[0] for e in top_entities]
    entity_freqs = [e[1] for e in top_entities]
    
    fig = px.bar(
        x=entity_names,
        y=entity_freqs,
        labels={'x': entity_type, 'y': 'Frequency'},
        color_discrete_sequence=['lightblue'],
    )
    
    fig.update_layout(
        title=f"Top {entity_type} Entities in News",
        xaxis_title=entity_type,
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
