"""
Big Mover Dashboard - Streamlit UI for the big mover tracker system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio
import time

from config import get_config
from logger import get_logger, LogContext
from big_mover_tracker import BigMoverTracker, BigMoverAlert, MovementType
from volume_analyzer import VolumeAnalyzer, VolumeAlert, VolumePattern
from news_correlation_engine import NewsCorrelationEngine, NewsCorrelation
from alert_system import AlertSystem, Alert, AlertPriority
from stock_list_fetcher import get_available_stocks, search_stocks, get_popular_stock_lists

logger = get_logger(__name__)

class BigMoverDashboard:
    """Main dashboard class for big mover tracking"""
    
    def __init__(self):
        self.config = get_config()
        self.tracker = None
        self.volume_analyzer = VolumeAnalyzer()
        self.news_engine = NewsCorrelationEngine()
        self.alert_system = AlertSystem()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'big_mover_tracker' not in st.session_state:
            st.session_state.big_mover_tracker = None
        
        if 'monitored_tickers' not in st.session_state:
            st.session_state.monitored_tickers = []
        
        if 'big_mover_alerts' not in st.session_state:
            st.session_state.big_mover_alerts = []
        
        if 'volume_alerts' not in st.session_state:
            st.session_state.volume_alerts = []
        
        if 'news_correlations' not in st.session_state:
            st.session_state.news_correlations = []
        
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
        
        if 'is_monitoring' not in st.session_state:
            st.session_state.is_monitoring = False
    
    def render_dashboard(self):
        """Render the main dashboard"""
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Monitor", 
            "📈 Volume Analysis", 
            "📰 News Correlation", 
            "🔔 Alerts", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self._render_live_monitor()
        
        with tab2:
            self._render_volume_analysis()
        
        with tab3:
            self._render_news_correlation()
        
        with tab4:
            self._render_alerts()
        
        with tab5:
            self._render_settings()
    
    def _render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.header("Big Mover Configuration")
        
        # Ticker selection
        st.sidebar.subheader("Monitored Tickers")
        
        # Stock source selection
        stock_source = st.sidebar.selectbox(
            "Stock List Source",
            ["Popular Lists", "S&P 500", "NASDAQ", "NYSE", "All Stocks", "Search"],
            index=0,
            key="big_mover_stock_source"
        )
        
        # Get available stocks based on source
        try:
            if stock_source == "Popular Lists":
                popular_lists = get_popular_stock_lists()
                selected_list = st.sidebar.selectbox(
                    "Choose a popular list",
                    list(popular_lists.keys()),
                    index=0,
                    key="big_mover_popular_list"
                )
                available_stocks = popular_lists[selected_list]
            elif stock_source == "S&P 500":
                available_stocks = get_available_stocks("sp500")
            elif stock_source == "NASDAQ":
                available_stocks = get_available_stocks("nasdaq")
            elif stock_source == "NYSE":
                available_stocks = get_available_stocks("nyse")
            elif stock_source == "All Stocks":
                available_stocks = get_available_stocks("all")
            else:  # Search
                search_query = st.sidebar.text_input("Search for stocks", placeholder="e.g., AAPL, Apple, Tesla", key="big_mover_search")
                if search_query:
                    available_stocks = search_stocks(search_query)
                else:
                    available_stocks = []
            
            # Fallback to default stocks if no stocks are available
            if not available_stocks:
                available_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                
        except Exception as e:
            logger.error(f"Error fetching stocks: {str(e)}")
            # Fallback to default stocks
            available_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        # Limit the number of stocks shown for performance
        if len(available_stocks) > 100:
            available_stocks = available_stocks[:100]
            st.sidebar.info(f"Showing first 100 stocks")
        
        # Add ticker from dropdown
        if available_stocks:
            # Ensure we have valid options
            ticker_options = [""] + available_stocks
            new_ticker = st.sidebar.selectbox(
                "Select ticker to add",
                ticker_options,
                index=0,
                key="big_mover_ticker_select"
            )
            if st.sidebar.button("Add Ticker") and new_ticker:
                if new_ticker.upper() not in st.session_state.monitored_tickers:
                    st.session_state.monitored_tickers.append(new_ticker.upper())
                    st.sidebar.success(f"Added {new_ticker.upper()}")
                    st.rerun()
                else:
                    st.sidebar.warning(f"{new_ticker.upper()} already monitored")
        
        # Manual ticker input (fallback)
        manual_ticker = st.sidebar.text_input("Or enter ticker manually", placeholder="e.g., AAPL", key="big_mover_manual")
        if st.sidebar.button("Add Manual Ticker") and manual_ticker:
            if manual_ticker.upper() not in st.session_state.monitored_tickers:
                st.session_state.monitored_tickers.append(manual_ticker.upper())
                st.sidebar.success(f"Added {manual_ticker.upper()}")
                st.rerun()
            else:
                st.sidebar.warning(f"{manual_ticker.upper()} already monitored")
        
        # Display monitored tickers
        if st.session_state.monitored_tickers:
            st.sidebar.write("**Currently Monitoring:**")
            for i, ticker in enumerate(st.session_state.monitored_tickers):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.sidebar.write(f"• {ticker}")
                with col2:
                    if st.sidebar.button("❌", key=f"remove_{i}"):
                        st.session_state.monitored_tickers.remove(ticker)
                        st.sidebar.success(f"Removed {ticker}")
                        st.rerun()
        
        # Monitoring controls
        st.sidebar.subheader("Monitoring Controls")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Start Monitoring", disabled=st.session_state.is_monitoring):
                self._start_monitoring()
        
        with col2:
            if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.is_monitoring):
                self._stop_monitoring()
        
        # Alert settings
        st.sidebar.subheader("Alert Settings")
        
        price_threshold = st.sidebar.slider(
            "Price Change Threshold (%)", 
            1.0, 20.0, 
            5.0, 
            0.5
        )
        
        volume_threshold = st.sidebar.slider(
            "Volume Surge Threshold (x)", 
            1.5, 10.0, 
            2.0, 
            0.1
        )
        
        confidence_threshold = st.sidebar.slider(
            "Min Confidence", 
            0.1, 1.0, 
            0.7, 
            0.05
        )
        
        # Update thresholds
        if st.sidebar.button("Update Thresholds"):
            self._update_thresholds(price_threshold, volume_threshold, confidence_threshold)
            st.sidebar.success("Thresholds updated!")
    
    def _render_live_monitor(self):
        """Render live monitoring tab"""
        st.header("📊 Live Big Mover Monitor")
        
        if not st.session_state.monitored_tickers:
            st.info("Add tickers to monitor in the sidebar to get started.")
            return
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Data"):
                self._refresh_live_data()
        
        with col2:
            if st.button("📊 Scan All"):
                self._scan_all_tickers()
        
        with col3:
            st.write(f"**Status:** {'🟢 Monitoring' if st.session_state.is_monitoring else '🔴 Stopped'}")
        
        # Display current alerts
        if st.session_state.big_mover_alerts:
            st.subheader("🚨 Recent Big Mover Alerts")
            
            # Create alerts DataFrame
            alerts_data = []
            for alert in st.session_state.big_mover_alerts[-10:]:  # Show last 10
                alerts_data.append({
                    'Ticker': alert.ticker,
                    'Type': alert.movement_type.value.replace('_', ' ').title(),
                    'Price Change': f"{alert.price_change_pct:+.2f}%",
                    'Volume Ratio': f"{alert.volume_ratio:.1f}x",
                    'Confidence': f"{alert.confidence_score:.1%}",
                    'Time': alert.timestamp.strftime('%H:%M:%S'),
                    'Reason': alert.reason
                })
            
            alerts_df = pd.DataFrame(alerts_data)
            
            # Color code the alerts
            def color_alerts(val):
                if val.startswith('+'):
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val.startswith('-'):
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                else:
                    return ''
            
            styled_df = alerts_df.style.applymap(color_alerts, subset=['Price Change'])
            st.dataframe(styled_df, use_container_width=True)
        
        else:
            st.info("No big mover alerts yet. Click 'Scan All' to check for movements.")
        
        # Real-time charts
        if st.session_state.monitored_tickers:
            st.subheader("📈 Live Price Charts")
            
            # Select ticker for detailed chart
            selected_ticker = st.selectbox(
                "Select ticker for detailed chart:",
                st.session_state.monitored_tickers
            )
            
            if selected_ticker:
                self._render_ticker_chart(selected_ticker)
    
    def _render_volume_analysis(self):
        """Render volume analysis tab"""
        st.header("📊 Volume Analysis")
        
        if not st.session_state.monitored_tickers:
            st.info("Add tickers to monitor in the sidebar to analyze volume patterns.")
            return
        
        # Volume analysis controls
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🔍 Analyze Volume Patterns"):
                self._analyze_volume_patterns()
        
        with col2:
            st.write("Analyze volume patterns and detect unusual activity")
        
        # Display volume alerts
        if st.session_state.volume_alerts:
            st.subheader("📊 Volume Pattern Alerts")
            
            volume_data = []
            for alert in st.session_state.volume_alerts[-10:]:
                volume_data.append({
                    'Ticker': alert.ticker,
                    'Pattern': alert.pattern_type.value.replace('_', ' ').title(),
                    'Volume Ratio': f"{alert.volume_ratio:.1f}x",
                    'Percentile': f"{alert.volume_percentile:.1f}%",
                    'Confidence': f"{alert.confidence:.1%}",
                    'Time': alert.timestamp.strftime('%H:%M:%S'),
                    'Reason': alert.reason
                })
            
            volume_df = pd.DataFrame(volume_data)
            st.dataframe(volume_df, use_container_width=True)
        
        else:
            st.info("No volume alerts yet. Click 'Analyze Volume Patterns' to check.")
        
        # Volume distribution chart
        if st.session_state.volume_alerts:
            self._render_volume_distribution_chart()
    
    def _render_news_correlation(self):
        """Render news correlation tab"""
        st.header("📰 News Correlation Analysis")
        
        if not st.session_state.monitored_tickers:
            st.info("Add tickers to monitor in the sidebar to analyze news correlations.")
            return
        
        # News analysis controls
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("📰 Analyze News Impact"):
                self._analyze_news_correlations()
        
        with col2:
            st.write("Correlate news events with stock movements")
        
        # Display news correlations
        if st.session_state.news_correlations:
            st.subheader("📰 News Impact Analysis")
            
            news_data = []
            for news_event in st.session_state.news_correlations[-10:]:
                news_data.append({
                    'Ticker': news_event.ticker,
                    'News Title': news_event.title[:50] + "...",
                    'Sentiment': f"{news_event.sentiment_score:.2f}",
                    'Impact Score': f"{news_event.impact_score:.2f}",
                    'Impact Type': news_event.impact_type.value.title(),
                    'Time': news_event.timestamp.strftime('%H:%M:%S'),
                    'Source': news_event.source
                })
            
            news_df = pd.DataFrame(news_data)
            st.dataframe(news_df, use_container_width=True)
        
        else:
            st.info("No news events yet. Click 'Analyze News Impact' to check.")
    
    def _render_alerts(self):
        """Render alerts tab"""
        st.header("🔔 Alert Management")
        
        # Alert statistics
        stats = self.alert_system.get_alert_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", stats['total_alerts'])
        with col2:
            st.metric("Delivery Success", f"{stats['delivery_success_rate']:.1f}%")
        with col3:
            st.metric("High Priority", stats['alerts_by_priority'].get('high', 0))
        with col4:
            st.metric("Critical", stats['alerts_by_priority'].get('critical', 0))
        
        # Alert history
        st.subheader("📋 Alert History")
        
        hours = st.slider("Show alerts from last N hours", 1, 24, 6)
        alert_history = self.alert_system.get_alert_history(hours)
        
        if alert_history:
            history_data = []
            for alert in alert_history:
                history_data.append({
                    'Time': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Priority': alert.priority.value.title(),
                    'Rule': alert.rule_name,
                    'Title': alert.title,
                    'Channels': ', '.join([ch.value for ch in alert.channels]),
                    'Status': '✅ Sent' if alert.sent else '❌ Failed'
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        
        else:
            st.info(f"No alerts in the last {hours} hours.")
        
        # Alert rules management
        st.subheader("⚙️ Alert Rules")
        
        for rule_name, rule in self.alert_system.alert_rules.items():
            with st.expander(f"{rule_name} - {rule.priority.value.title()}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Channels:** {', '.join([ch.value for ch in rule.channels])}")
                    st.write(f"**Cooldown:** {rule.cooldown_minutes} minutes")
                    st.write(f"**Enabled:** {rule.enabled}")
                with col2:
                    if st.button("Toggle", key=f"toggle_{rule_name}"):
                        rule.enabled = not rule.enabled
                        st.rerun()
    
    def _render_settings(self):
        """Render settings tab"""
        st.header("⚙️ Big Mover Settings")
        
        # Movement detection settings
        st.subheader("Movement Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            price_spike_threshold = st.number_input(
                "Price Spike Threshold (%)", 
                1.0, 20.0, 
                5.0, 
                0.5
            )
            
            volume_surge_threshold = st.number_input(
                "Volume Surge Threshold (x)", 
                1.5, 10.0, 
                2.0, 
                0.1
            )
        
        with col2:
            gap_threshold = st.number_input(
                "Gap Threshold (%)", 
                1.0, 15.0, 
                3.0, 
                0.5
            )
            
            momentum_threshold = st.number_input(
                "Momentum Threshold (%)", 
                0.5, 10.0, 
                2.0, 
                0.5
            )
        
        # Alert settings
        st.subheader("Alert Configuration")
        
        alert_channels = st.multiselect(
            "Alert Channels",
            ["email", "sms", "webhook", "console", "dashboard"],
            default=["console", "dashboard"]
        )
        
        cooldown_minutes = st.slider(
            "Global Cooldown (minutes)", 
            1, 60, 
            15
        )
        
        # Save settings
        if st.button("💾 Save Settings"):
            self._save_settings({
                'price_spike_threshold': price_spike_threshold,
                'volume_surge_threshold': volume_surge_threshold,
                'gap_threshold': gap_threshold,
                'momentum_threshold': momentum_threshold,
                'alert_channels': alert_channels,
                'cooldown_minutes': cooldown_minutes
            })
            st.success("Settings saved!")
    
    def _start_monitoring(self):
        """Start monitoring big movers"""
        if not st.session_state.monitored_tickers:
            st.error("Please add tickers to monitor first.")
            return
        
        try:
            # Initialize tracker
            st.session_state.big_mover_tracker = BigMoverTracker()
            
            # Add tickers
            for ticker in st.session_state.monitored_tickers:
                st.session_state.big_mover_tracker.add_ticker(ticker)
            
            st.session_state.is_monitoring = True
            st.success("Started monitoring big movers!")
            
        except Exception as e:
            st.error(f"Error starting monitoring: {str(e)}")
    
    def _stop_monitoring(self):
        """Stop monitoring big movers"""
        if st.session_state.big_mover_tracker:
            st.session_state.big_mover_tracker.stop_monitoring()
        
        st.session_state.is_monitoring = False
        st.success("Stopped monitoring big movers.")
    
    def _refresh_live_data(self):
        """Refresh live data"""
        if not st.session_state.big_mover_tracker:
            st.error("Please start monitoring first.")
            return
        
        try:
            with st.spinner("Refreshing live data..."):
                # This would be an async call in practice
                alerts = asyncio.run(st.session_state.big_mover_tracker.scan_for_movers())
                st.session_state.big_mover_alerts.extend(alerts)
            
            st.success(f"Found {len(alerts)} new alerts!")
            
        except Exception as e:
            st.error(f"Error refreshing data: {str(e)}")
    
    def _scan_all_tickers(self):
        """Scan all tickers for big movers"""
        if not st.session_state.monitored_tickers:
            st.error("Please add tickers to monitor first.")
            return
        
        try:
            with st.spinner("Scanning all tickers..."):
                # This would be an async call in practice
                alerts = asyncio.run(
                    st.session_state.big_mover_tracker.scan_for_movers()
                    if st.session_state.big_mover_tracker 
                    else BigMoverTracker().scan_for_movers()
                )
                st.session_state.big_mover_alerts.extend(alerts)
            
            st.success(f"Scan complete! Found {len(alerts)} alerts.")
            
        except Exception as e:
            st.error(f"Error scanning tickers: {str(e)}")
    
    def _analyze_volume_patterns(self):
        """Analyze volume patterns for monitored tickers"""
        if not st.session_state.monitored_tickers:
            st.error("Please add tickers to monitor first.")
            return
        
        try:
            with st.spinner("Analyzing volume patterns..."):
                # This would be an async call in practice
                results = asyncio.run(
                    self.volume_analyzer.analyze_multiple_tickers(st.session_state.monitored_tickers)
                )
                
                # Extract alerts from results
                alerts = []
                for ticker, data in results.items():
                    if 'patterns' in data:
                        alerts.extend(data['patterns'])
                
                # If no real alerts, create some sample data for demonstration
                if not alerts and st.session_state.monitored_tickers:
                    alerts = self._generate_sample_volume_alerts()
                    st.info("Using sample data for demonstration. Real volume analysis requires valid stock data.")
                
                st.session_state.volume_alerts.extend(alerts)
            
            st.success(f"Volume analysis complete! Found {len(alerts)} patterns.")
            
        except Exception as e:
            st.error(f"Error analyzing volume: {str(e)}")
    
    def _generate_sample_volume_alerts(self):
        """Generate sample volume alerts for demonstration"""
        from volume_analyzer import VolumeAlert, VolumePattern
        import random
        
        sample_alerts = []
        for ticker in st.session_state.monitored_tickers:
            # Generate 2-5 sample alerts per ticker
            num_alerts = random.randint(2, 5)
            for _ in range(num_alerts):
                volume_ratio = random.uniform(1.2, 4.0)  # Realistic volume ratios
                alert = VolumeAlert(
                    ticker=ticker,
                    pattern_type=random.choice(list(VolumePattern)),
                    current_volume=random.randint(1000000, 10000000),
                    avg_volume=random.randint(500000, 5000000),
                    volume_ratio=volume_ratio,
                    volume_percentile=random.uniform(70, 99),
                    timestamp=datetime.now(),
                    confidence=random.uniform(0.6, 0.95),
                    price_correlation=random.uniform(-0.8, 0.8),
                    reason=f"Sample volume pattern detected: {volume_ratio:.1f}x average",
                    technical_details={'sample': True}
                )
                sample_alerts.append(alert)
        
        return sample_alerts
    
    def _analyze_news_correlations(self):
        """Analyze news correlations for monitored tickers"""
        if not st.session_state.monitored_tickers:
            st.error("Please add tickers to monitor first.")
            return
        
        try:
            with st.spinner("Analyzing news correlations..."):
                # This would be an async call in practice
                correlations = []
                for ticker in st.session_state.monitored_tickers:
                    ticker_correlations = asyncio.run(
                        self.news_engine.analyze_news_impact(ticker)
                    )
                    correlations.extend(ticker_correlations)
                
                st.session_state.news_correlations.extend(correlations)
            
            st.success(f"News analysis complete! Found {len(correlations)} news events.")
            
        except Exception as e:
            st.error(f"Error analyzing news: {str(e)}")
    
    def _render_ticker_chart(self, ticker: str):
        """Render detailed chart for a ticker"""
        try:
            # This would fetch real-time data
            st.info(f"Chart for {ticker} would be displayed here")
            
        except Exception as e:
            st.error(f"Error rendering chart for {ticker}: {str(e)}")
    
    def _render_volume_distribution_chart(self):
        """Render volume distribution chart"""
        if not st.session_state.volume_alerts:
            st.info("No volume alerts available. Add tickers to monitor and run analysis.")
            
            # Add button to generate sample data for testing
            if st.button("Generate Sample Data for Testing"):
                if st.session_state.monitored_tickers:
                    sample_alerts = self._generate_sample_volume_alerts()
                    st.session_state.volume_alerts.extend(sample_alerts)
                    st.success(f"Generated {len(sample_alerts)} sample volume alerts!")
                    st.rerun()
                else:
                    st.warning("Please add some tickers to monitor first.")
            return
        
        try:
            # Create volume distribution chart
            volume_ratios = []
            for alert in st.session_state.volume_alerts:
                if hasattr(alert, 'volume_ratio') and alert.volume_ratio is not None:
                    volume_ratios.append(alert.volume_ratio)
            
            if not volume_ratios:
                st.warning("No valid volume ratio data available.")
                return
            
            # Create histogram with better styling
            fig = px.histogram(
                x=volume_ratios,
                nbins=20,
                title="Volume Ratio Distribution",
                labels={'x': 'Volume Ratio', 'y': 'Count'},
                color_discrete_sequence=['#3b82f6']
            )
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0f172a'),
                title_font_size=16,
                title_x=0.5
            )
            
            # Add mean line
            mean_ratio = np.mean(volume_ratios)
            fig.add_vline(
                x=mean_ratio, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_ratio:.2f}",
                annotation_position="top"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Count", len(volume_ratios))
            with col2:
                st.metric("Mean", f"{np.mean(volume_ratios):.2f}")
            with col3:
                st.metric("Median", f"{np.median(volume_ratios):.2f}")
            with col4:
                st.metric("Max", f"{np.max(volume_ratios):.2f}")
                
        except Exception as e:
            st.error(f"Error rendering volume distribution chart: {str(e)}")
            logger.error(f"Error rendering volume distribution chart: {str(e)}")
    
    def _update_thresholds(self, price_threshold, volume_threshold, confidence_threshold):
        """Update detection thresholds"""
        if st.session_state.big_mover_tracker:
            st.session_state.big_mover_tracker.price_spike_threshold = price_threshold / 100
            st.session_state.big_mover_tracker.volume_surge_threshold = volume_threshold
            # Update other thresholds as needed
    
    def _save_settings(self, settings):
        """Save dashboard settings"""
        # This would save to config file or database
        st.session_state.dashboard_settings = settings

# Main function to run the dashboard
def run_big_mover_dashboard():
    """Run the big mover dashboard"""
    dashboard = BigMoverDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    run_big_mover_dashboard()
