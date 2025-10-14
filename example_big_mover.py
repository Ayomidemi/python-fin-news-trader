"""
Example script demonstrating how to use the Big Mover Tracker system
"""

import asyncio
import time
from datetime import datetime
from big_mover_tracker import BigMoverTracker, create_big_mover_tracker
from volume_analyzer import VolumeAnalyzer, analyze_ticker_volume
from news_correlation_engine import NewsCorrelationEngine, analyze_news_impact
from alert_system import AlertSystem, create_alert_system

async def example_big_mover_scan():
    """Example of scanning for big movers"""
    print("🚀 Big Mover Tracker Example")
    print("=" * 50)
    
    # Create tracker
    tracker = create_big_mover_tracker()
    
    # Add some popular tickers to monitor
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    print(f"Monitoring tickers: {', '.join(tickers)}")
    
    # Add tickers to tracker
    for ticker in tickers:
        tracker.add_ticker(ticker)
    
    # Scan for big movers
    print("\n🔍 Scanning for big movers...")
    alerts = await tracker.scan_for_movers()
    
    if alerts:
        print(f"\n📊 Found {len(alerts)} big mover alerts:")
        for alert in alerts:
            print(f"  • {alert.ticker}: {alert.movement_type.value} - {alert.reason}")
            print(f"    Price: ${alert.current_price:.2f} ({alert.price_change_pct:+.2f}%)")
            print(f"    Volume: {alert.volume:,} ({alert.volume_ratio:.1f}x avg)")
            print(f"    Confidence: {alert.confidence_score:.1%}")
            print()
    else:
        print("No big movers detected at this time.")
    
    return alerts

async def example_volume_analysis():
    """Example of volume analysis"""
    print("\n📊 Volume Analysis Example")
    print("=" * 50)
    
    # Analyze volume for a specific ticker
    ticker = "AAPL"
    print(f"Analyzing volume patterns for {ticker}...")
    
    try:
        analysis = await analyze_ticker_volume(ticker)
        
        if analysis and 'patterns' in analysis:
            patterns = analysis['patterns']
            print(f"Found {len(patterns)} volume patterns:")
            
            for pattern in patterns:
                print(f"  • {pattern.pattern_type.value}: {pattern.reason}")
                print(f"    Volume: {pattern.current_volume:,} ({pattern.volume_ratio:.1f}x avg)")
                print(f"    Confidence: {pattern.confidence:.1%}")
                print()
        else:
            print("No significant volume patterns detected.")
    
    except Exception as e:
        print(f"Error analyzing volume: {e}")

async def example_news_correlation():
    """Example of news correlation analysis"""
    print("\n📰 News Correlation Example")
    print("=" * 50)
    
    # Analyze news impact for a specific ticker
    ticker = "TSLA"
    print(f"Analyzing news impact for {ticker}...")
    
    try:
        news_events = await analyze_news_impact(ticker)
        
        if news_events:
            print(f"Found {len(news_events)} news events:")
            
            for event in news_events[:3]:  # Show top 3
                print(f"  • {event.title[:60]}...")
                print(f"    Sentiment: {event.sentiment_score:.2f}")
                print(f"    Impact: {event.impact_score:.2f} ({event.impact_type.value})")
                print(f"    Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("No recent news events found.")
    
    except Exception as e:
        print(f"Error analyzing news: {e}")

async def example_alert_system():
    """Example of alert system"""
    print("\n🔔 Alert System Example")
    print("=" * 50)
    
    # Create alert system
    alert_system = create_alert_system()
    
    # Get alert statistics
    stats = alert_system.get_alert_stats()
    print(f"Alert Statistics:")
    print(f"  Total Alerts: {stats['total_alerts']}")
    print(f"  Delivery Success Rate: {stats['delivery_success_rate']:.1f}%")
    print(f"  Alerts by Priority: {stats['alerts_by_priority']}")
    
    # Get recent alerts
    recent_alerts = alert_system.get_alert_history(hours=24)
    print(f"\nRecent Alerts (last 24h): {len(recent_alerts)}")
    
    for alert in recent_alerts[:3]:  # Show last 3
        print(f"  • {alert.title}")
        print(f"    Priority: {alert.priority.value}")
        print(f"    Time: {alert.timestamp.strftime('%H:%M:%S')}")
        print()

async def example_continuous_monitoring():
    """Example of continuous monitoring"""
    print("\n🔄 Continuous Monitoring Example")
    print("=" * 50)
    
    # Create tracker
    tracker = create_big_mover_tracker(["AAPL", "MSFT", "GOOGL"])
    
    print("Starting continuous monitoring (will run for 30 seconds)...")
    print("Press Ctrl+C to stop early")
    
    try:
        # Start monitoring in background
        monitoring_task = asyncio.create_task(
            tracker.start_monitoring(scan_interval=10)  # Scan every 10 seconds
        )
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop monitoring
        tracker.stop_monitoring()
        monitoring_task.cancel()
        
        print("\nMonitoring stopped.")
        
        # Show any alerts that were generated
        recent_alerts = tracker.get_recent_alerts(hours=1)
        if recent_alerts:
            print(f"Generated {len(recent_alerts)} alerts during monitoring:")
            for alert in recent_alerts:
                print(f"  • {alert.ticker}: {alert.movement_type.value}")
        else:
            print("No alerts generated during monitoring period.")
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        tracker.stop_monitoring()

async def main():
    """Main example function"""
    print("🚀 FinNewsTrader Big Mover Tracker Examples")
    print("=" * 60)
    
    # Run examples
    await example_big_mover_scan()
    await example_volume_analysis()
    await example_news_correlation()
    await example_alert_system()
    
    # Ask user if they want to try continuous monitoring
    print("\n" + "=" * 60)
    print("Would you like to try continuous monitoring? (y/n)")
    
    # For demo purposes, we'll skip the continuous monitoring
    # Uncomment the line below to enable it
    # await example_continuous_monitoring()
    
    print("\n✅ Examples completed!")
    print("\nTo use the Big Mover Tracker in your app:")
    print("1. Run: streamlit run app.py")
    print("2. Go to the 'Big Mover Tracker' tab")
    print("3. Add tickers to monitor")
    print("4. Start monitoring and watch for alerts!")

if __name__ == "__main__":
    asyncio.run(main())
