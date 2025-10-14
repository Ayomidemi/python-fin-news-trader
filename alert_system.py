"""
Alert System - Real-time notifications for big movers and trading opportunities
"""

import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from config import get_config
from logger import get_logger, LogContext
from big_mover_tracker import BigMoverAlert, MovementType
from volume_analyzer import VolumeAlert, VolumePattern
from news_correlation_engine import NewsCorrelation, NewsEvent

logger = get_logger(__name__)

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    DASHBOARD = "dashboard"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    enabled: bool
    priority: AlertPriority
    channels: List[AlertChannel]
    conditions: Dict[str, Any]
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    rule_name: str
    priority: AlertPriority
    channels: List[AlertChannel]
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    sent: bool = False
    delivery_status: Dict[str, str] = None

class AlertSystem:
    """Main alert system for big mover notifications"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.alert_rules = {}
        self.alert_history = []
        self.delivery_handlers = {}
        
        # Initialize delivery handlers
        self._initialize_delivery_handlers()
        
        # Load default alert rules
        self._load_default_rules()
    
    def _initialize_delivery_handlers(self):
        """Initialize delivery channel handlers"""
        self.delivery_handlers = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.CONSOLE: self._send_console_alert,
            AlertChannel.DASHBOARD: self._send_dashboard_alert
        }
    
    def _load_default_rules(self):
        """Load default alert rules"""
        # Price spike rule
        self.add_alert_rule(AlertRule(
            name="price_spike",
            enabled=True,
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
            conditions={
                "movement_type": MovementType.PRICE_SPIKE,
                "min_price_change_pct": 5.0,
                "min_confidence": 0.7
            },
            cooldown_minutes=15
        ))
        
        # Volume surge rule
        self.add_alert_rule(AlertRule(
            name="volume_surge",
            enabled=True,
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
            conditions={
                "pattern_type": VolumePattern.SURGE,
                "min_volume_ratio": 2.0,
                "min_confidence": 0.6
            },
            cooldown_minutes=30
        ))
        
        # Gap up rule
        self.add_alert_rule(AlertRule(
            name="gap_up",
            enabled=True,
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.CONSOLE],
            conditions={
                "movement_type": MovementType.GAP_UP,
                "min_price_change_pct": 3.0,
                "min_confidence": 0.8
            },
            cooldown_minutes=10
        ))
        
        # Breakout rule
        self.add_alert_rule(AlertRule(
            name="breakout",
            enabled=True,
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
            conditions={
                "movement_type": MovementType.BREAKOUT,
                "min_confidence": 0.6
            },
            cooldown_minutes=20
        ))
        
        # High impact news rule
        self.add_alert_rule(AlertRule(
            name="high_impact_news",
            enabled=True,
            priority=AlertPriority.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.CONSOLE],
            conditions={
                "news_impact_type": "high_impact",
                "min_correlation_strength": 0.7
            },
            cooldown_minutes=5
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def update_alert_rule(self, rule_name: str, **kwargs):
        """Update an alert rule"""
        if rule_name in self.alert_rules:
            rule = self.alert_rules[rule_name]
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info(f"Updated alert rule: {rule_name}")
    
    async def process_big_mover_alert(self, alert: BigMoverAlert) -> List[Alert]:
        """Process a big mover alert and generate notifications"""
        generated_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            # Check if rule matches
            if self._matches_rule(alert, rule):
                # Generate alert
                alert_obj = self._create_alert_from_big_mover(alert, rule)
                generated_alerts.append(alert_obj)
                
                # Update rule last triggered
                rule.last_triggered = datetime.now()
        
        # Send alerts
        for alert_obj in generated_alerts:
            await self._send_alert(alert_obj)
        
        return generated_alerts
    
    async def process_volume_alert(self, alert: VolumeAlert) -> List[Alert]:
        """Process a volume alert and generate notifications"""
        generated_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            # Check if rule matches
            if self._matches_volume_rule(alert, rule):
                # Generate alert
                alert_obj = self._create_alert_from_volume(alert, rule)
                generated_alerts.append(alert_obj)
                
                # Update rule last triggered
                rule.last_triggered = datetime.now()
        
        # Send alerts
        for alert_obj in generated_alerts:
            await self._send_alert(alert_obj)
        
        return generated_alerts
    
    async def process_news_correlation(self, correlation: NewsCorrelation) -> List[Alert]:
        """Process a news correlation and generate notifications"""
        generated_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            # Check if rule matches
            if self._matches_news_rule(correlation, rule):
                # Generate alert
                alert_obj = self._create_alert_from_news_correlation(correlation, rule)
                generated_alerts.append(alert_obj)
                
                # Update rule last triggered
                rule.last_triggered = datetime.now()
        
        # Send alerts
        for alert_obj in generated_alerts:
            await self._send_alert(alert_obj)
        
        return generated_alerts
    
    def _matches_rule(self, alert: BigMoverAlert, rule: AlertRule) -> bool:
        """Check if a big mover alert matches a rule"""
        conditions = rule.conditions
        
        # Check movement type
        if 'movement_type' in conditions:
            if alert.movement_type.value != conditions['movement_type'].value:
                return False
        
        # Check price change percentage
        if 'min_price_change_pct' in conditions:
            if abs(alert.price_change_pct) < conditions['min_price_change_pct']:
                return False
        
        # Check confidence
        if 'min_confidence' in conditions:
            if alert.confidence_score < conditions['min_confidence']:
                return False
        
        return True
    
    def _matches_volume_rule(self, alert: VolumeAlert, rule: AlertRule) -> bool:
        """Check if a volume alert matches a rule"""
        conditions = rule.conditions
        
        # Check pattern type
        if 'pattern_type' in conditions:
            if alert.pattern_type.value != conditions['pattern_type'].value:
                return False
        
        # Check volume ratio
        if 'min_volume_ratio' in conditions:
            if alert.volume_ratio < conditions['min_volume_ratio']:
                return False
        
        # Check confidence
        if 'min_confidence' in conditions:
            if alert.confidence < conditions['min_confidence']:
                return False
        
        return True
    
    def _matches_news_rule(self, correlation: NewsCorrelation, rule: AlertRule) -> bool:
        """Check if a news correlation matches a rule"""
        conditions = rule.conditions
        
        # Check news impact type
        if 'news_impact_type' in conditions:
            if correlation.news_event.impact_type.value != conditions['news_impact_type']:
                return False
        
        # Check correlation strength
        if 'min_correlation_strength' in conditions:
            if correlation.correlation_strength < conditions['min_correlation_strength']:
                return False
        
        return True
    
    def _create_alert_from_big_mover(self, alert: BigMoverAlert, rule: AlertRule) -> Alert:
        """Create an alert from a big mover alert"""
        alert_id = f"big_mover_{alert.ticker}_{int(datetime.now().timestamp())}"
        
        title = f"🚨 Big Mover Alert: {alert.ticker}"
        message = self._format_big_mover_message(alert)
        
        data = {
            'type': 'big_mover',
            'ticker': alert.ticker,
            'movement_type': alert.movement_type.value,
            'price_change_pct': alert.price_change_pct,
            'volume_ratio': alert.volume_ratio,
            'confidence': alert.confidence_score,
            'reason': alert.reason
        }
        
        return Alert(
            id=alert_id,
            rule_name=rule.name,
            priority=rule.priority,
            channels=rule.channels,
            title=title,
            message=message,
            data=data,
            timestamp=datetime.now()
        )
    
    def _create_alert_from_volume(self, alert: VolumeAlert, rule: AlertRule) -> Alert:
        """Create an alert from a volume alert"""
        alert_id = f"volume_{alert.ticker}_{int(datetime.now().timestamp())}"
        
        title = f"📊 Volume Alert: {alert.ticker}"
        message = self._format_volume_message(alert)
        
        data = {
            'type': 'volume',
            'ticker': alert.ticker,
            'pattern_type': alert.pattern_type.value,
            'volume_ratio': alert.volume_ratio,
            'confidence': alert.confidence,
            'reason': alert.reason
        }
        
        return Alert(
            id=alert_id,
            rule_name=rule.name,
            priority=rule.priority,
            channels=rule.channels,
            title=title,
            message=message,
            data=data,
            timestamp=datetime.now()
        )
    
    def _create_alert_from_news_correlation(self, correlation: NewsCorrelation, rule: AlertRule) -> Alert:
        """Create an alert from a news correlation"""
        alert_id = f"news_{correlation.ticker}_{int(datetime.now().timestamp())}"
        
        title = f"📰 News Impact: {correlation.ticker}"
        message = self._format_news_correlation_message(correlation)
        
        data = {
            'type': 'news_correlation',
            'ticker': correlation.ticker,
            'news_title': correlation.news_event.title,
            'price_movement_pct': correlation.price_movement_pct,
            'correlation_strength': correlation.correlation_strength,
            'confidence': correlation.confidence
        }
        
        return Alert(
            id=alert_id,
            rule_name=rule.name,
            priority=rule.priority,
            channels=rule.channels,
            title=title,
            message=message,
            data=data,
            timestamp=datetime.now()
        )
    
    def _format_big_mover_message(self, alert: BigMoverAlert) -> str:
        """Format big mover alert message"""
        direction = "📈" if alert.price_change_pct > 0 else "📉"
        
        message = f"""
{direction} {alert.ticker} - {alert.movement_type.value.replace('_', ' ').title()}

Price: ${alert.current_price:.2f} ({alert.price_change_pct:+.2f}%)
Volume: {alert.volume:,} ({alert.volume_ratio:.1f}x average)
Confidence: {alert.confidence_score:.1%}
Reason: {alert.reason}

Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return message
    
    def _format_volume_message(self, alert: VolumeAlert) -> str:
        """Format volume alert message"""
        message = f"""
📊 {alert.ticker} - {alert.pattern_type.value.replace('_', ' ').title()}

Volume: {alert.current_volume:,} ({alert.volume_ratio:.1f}x average)
Percentile: {alert.volume_percentile:.1f}%
Confidence: {alert.confidence:.1%}
Reason: {alert.reason}

Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return message
    
    def _format_news_correlation_message(self, correlation: NewsCorrelation) -> str:
        """Format news correlation message"""
        direction = "📈" if correlation.price_movement_pct > 0 else "📉"
        
        message = f"""
{direction} {correlation.ticker} - News Impact

News: {correlation.news_event.title[:100]}...
Price Movement: {correlation.price_movement_pct:+.2f}%
Correlation: {correlation.correlation_strength:.1%}
Confidence: {correlation.confidence:.1%}

Analysis: {correlation.impact_analysis}

Time: {correlation.news_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return message
    
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        delivery_status = {}
        
        for channel in alert.channels:
            try:
                handler = self.delivery_handlers.get(channel)
                if handler:
                    status = await handler(alert)
                    delivery_status[channel.value] = status
                else:
                    delivery_status[channel.value] = "no_handler"
            except Exception as e:
                logger.error(f"Error sending alert via {channel.value}: {str(e)}")
                delivery_status[channel.value] = f"error: {str(e)}"
        
        alert.delivery_status = delivery_status
        alert.sent = True
        self.alert_history.append(alert)
        
        logger.info(f"Alert {alert.id} sent via {len(alert.channels)} channels")
    
    async def _send_email_alert(self, alert: Alert) -> str:
        """Send email alert"""
        # This is a placeholder - you'd need to configure SMTP settings
        logger.info(f"Email alert sent: {alert.title}")
        return "sent"
    
    async def _send_sms_alert(self, alert: Alert) -> str:
        """Send SMS alert"""
        # This is a placeholder - you'd need to integrate with SMS service
        logger.info(f"SMS alert sent: {alert.title}")
        return "sent"
    
    async def _send_webhook_alert(self, alert: Alert) -> str:
        """Send webhook alert"""
        # This is a placeholder - you'd need to configure webhook URLs
        logger.info(f"Webhook alert sent: {alert.title}")
        return "sent"
    
    async def _send_console_alert(self, alert: Alert) -> str:
        """Send console alert"""
        print(f"\n{'='*50}")
        print(f"ALERT: {alert.title}")
        print(f"{'='*50}")
        print(alert.message)
        print(f"{'='*50}\n")
        return "sent"
    
    async def _send_dashboard_alert(self, alert: Alert) -> str:
        """Send dashboard alert"""
        # This would integrate with your dashboard system
        logger.info(f"Dashboard alert sent: {alert.title}")
        return "sent"
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        
        if total_alerts == 0:
            return {
                'total_alerts': 0,
                'alerts_by_priority': {},
                'alerts_by_rule': {},
                'delivery_success_rate': 0
            }
        
        # Alerts by priority
        priority_counts = {}
        for alert in self.alert_history:
            priority = alert.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Alerts by rule
        rule_counts = {}
        for alert in self.alert_history:
            rule = alert.rule_name
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Delivery success rate
        successful_deliveries = 0
        total_deliveries = 0
        
        for alert in self.alert_history:
            if alert.delivery_status:
                for channel, status in alert.delivery_status.items():
                    total_deliveries += 1
                    if status == "sent":
                        successful_deliveries += 1
        
        delivery_success_rate = (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
        
        return {
            'total_alerts': total_alerts,
            'alerts_by_priority': priority_counts,
            'alerts_by_rule': rule_counts,
            'delivery_success_rate': delivery_success_rate
        }

# Convenience functions
def create_alert_system() -> AlertSystem:
    """Create a new alert system instance"""
    return AlertSystem()

async def send_big_mover_alert(alert: BigMoverAlert) -> List[Alert]:
    """Send a big mover alert"""
    system = AlertSystem()
    return await system.process_big_mover_alert(alert)
