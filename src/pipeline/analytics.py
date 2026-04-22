"""
Module 4: Analytics Engine
Processes historical usage data to generate insights, usage summaries, 
and cost estimations for the EMS dashboard.
"""
import datetime

class AnalyticsEngine:
    def __init__(self, cost_per_kwh: float = 0.15):
        """
        Initialize the Analytics Engine.
        
        Args:
            cost_per_kwh: The electricity tariff in dollars per kWh.
        """
        self.cost_per_kwh = cost_per_kwh
        # In-memory storage for simplicity in this module representation
        self.daily_usage = {}
        
    def record_usage(self, device_id: str, watts: float, duration_hours: float):
        """
        Record power usage for a device.
        
        Args:
            device_id: Identifier for the device.
            watts: Power consumption in watts.
            duration_hours: Duration the device was active in hours.
        """
        today = datetime.date.today().isoformat()
        if today not in self.daily_usage:
            self.daily_usage[today] = {}
            
        if device_id not in self.daily_usage[today]:
            self.daily_usage[today][device_id] = 0.0
            
        # Convert to kWh
        kwh = (watts * duration_hours) / 1000.0
        self.daily_usage[today][device_id] += kwh
        
    def get_daily_summary(self, date_str: str = None) -> dict:
        """
        Get the usage summary and cost for a specific day.
        
        Args:
            date_str: Date string in ISO format (YYYY-MM-DD). Defaults to today.
            
        Returns:
            dict: Summary of usage per device and total cost.
        """
        if date_str is None:
            date_str = datetime.date.today().isoformat()
            
        day_data = self.daily_usage.get(date_str, {})
        total_kwh = sum(day_data.values())
        total_cost = total_kwh * self.cost_per_kwh
        
        return {
            "date": date_str,
            "device_usage_kwh": day_data,
            "total_kwh": round(total_kwh, 3),
            "estimated_cost_usd": round(total_cost, 2)
        }

analytics_engine = AnalyticsEngine()
