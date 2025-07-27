import time
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AlertSystem:
    """Real-time alert system with logging"""
    
    def __init__(self):
        self.alert_cooldown = {}
        self.cooldown_duration = 10  # 10 seconds between same alerts
        self.alert_logs = deque(maxlen=100)  # Keep last 100 alerts
        
    def should_send_alert(self, alert_type):
        """Check if alert should be sent (cooldown management)"""
        now = time.time()
        if alert_type in self.alert_cooldown:
            if now - self.alert_cooldown[alert_type] < self.cooldown_duration:
                return False
        
        self.alert_cooldown[alert_type] = now
        return True
    
    def log_alert(self, alert_type, message):
        """Log alert with timestamp"""
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message
        }
        self.alert_logs.append(alert_entry)
        logger.info(f"ALERT: {alert_type} - {message}")
    
    def trigger_crowd_alert(self, count, timestamp):
        """Trigger crowd detection alert"""
        if self.should_send_alert('crowd'):
            message = f"Crowd of {count} people detected"
            self.log_alert('crowd', message)
            return True
        return False
    
    def trigger_inactivity_alert(self, object_id, duration, timestamp):
        """Trigger inactivity alert"""
        if self.should_send_alert(f'inactivity_{object_id}'):
            message = f"Person ID {object_id} inactive for {duration:.1f} minutes"
            self.log_alert('inactivity', message)
            return True
        return False
    
    def get_recent_alerts(self, limit=10):
        """Get recent alerts for display"""
        return list(self.alert_logs)[-limit:]