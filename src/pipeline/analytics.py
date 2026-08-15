"""
Module 4: Analytics Engine
Processes historical usage data to generate insights, usage summaries, 
and cost estimations for the EMS dashboard.
"""
from typing import Dict, Optional, Union
import datetime
import os
import yaml


def compute_tou_cost(
    watts: float,
    seconds: float = 1.0,
    period: str = "peak",
    rate: Optional[float] = None
) -> float:
    """
    Compute electricity cost using Time-of-Use (ToU) rates.
    
    Args:
        watts: Power consumption in Watts.
        seconds: Duration of consumption in seconds.
        period: Pricing window ("peak", "mid", "off-peak").
        rate: Optional explicit rate override ($/kWh).
        
    Returns:
        float: Cost in USD.
    """
    if rate is None:
        p = period.lower().replace("-", "_").replace(" ", "_")
        if "peak" in p and "off" not in p and "mid" not in p:
            rate = 0.28
        elif "mid" in p:
            rate = 0.18
        elif "off" in p:
            rate = 0.09
        else:
            rate = 0.15

    kwh = (watts / 1000.0) * (seconds / 3600.0)
    return kwh * rate


class AnalyticsEngine:
    def __init__(
        self,
        cost_per_kwh: float = 0.15,
        config: Optional[dict] = None,
        config_path: Optional[str] = "config/config.yaml"
    ):
        """
        Initialize the Analytics Engine.
        
        Args:
            cost_per_kwh: Fallback electricity tariff in dollars per kWh.
            config: Optional configuration dictionary.
            config_path: Path to YAML config file.
        """
        self.cost_per_kwh = cost_per_kwh
        self.config = config or {}
        
        if not self.config and config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception:
                self.config = {}

        # Load ToU pricing configuration
        analytics_cfg = self.config.get("analytics", {})
        self.tou_pricing = analytics_cfg.get("tou_pricing", {
            "peak": {"hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "rate": 0.28},
            "mid": {"hours": [7, 8, 21, 22], "rate": 0.18},
            "off_peak": {"hours": [0, 1, 2, 3, 4, 5, 6, 23], "rate": 0.09},
        })

        # In-memory daily storage: {date_iso: {device_id: value}}
        self.daily_usage: Dict[str, Dict[str, float]] = {}
        self.daily_cost: Dict[str, Dict[str, float]] = {}

    def get_tou_rate(self, hour_or_dt: Union[datetime.datetime, int, None] = None) -> float:
        """
        Get the Time-of-Use rate for a given hour or datetime.
        """
        if hour_or_dt is None:
            hour = datetime.datetime.now().hour
        elif isinstance(hour_or_dt, datetime.datetime):
            hour = hour_or_dt.hour
        else:
            hour = int(hour_or_dt)

        for tier, data in self.tou_pricing.items():
            if hour in data.get("hours", []):
                return data.get("rate", self.cost_per_kwh)
        return self.cost_per_kwh

    async def record(
        self,
        device_id: str,
        watts: float,
        seconds: float = 1.0,
        timestamp: Optional[datetime.datetime] = None
    ) -> float:
        """
        Record power reading asynchronously (typically 1Hz).
        
        Args:
            device_id: Identifier for the device.
            watts: Power consumption in Watts.
            seconds: Reading duration (default 1.0s).
            timestamp: Optional timestamp for the reading.
            
        Returns:
            float: kWh accumulated in this reading.
        """
        dt = timestamp or datetime.datetime.now()
        today = dt.date().isoformat()

        if today not in self.daily_usage:
            self.daily_usage[today] = {}
        if today not in self.daily_cost:
            self.daily_cost[today] = {}

        kwh = (watts / 1000.0) * (seconds / 3600.0)
        rate = self.get_tou_rate(dt.hour)
        cost = kwh * rate

        self.daily_usage[today][device_id] = self.daily_usage[today].get(device_id, 0.0) + kwh
        self.daily_cost[today][device_id] = self.daily_cost[today].get(device_id, 0.0) + cost

        return kwh

    def record_usage(
        self,
        device_id: str,
        watts: float,
        duration_hours: float,
        date_str: Optional[str] = None
    ):
        """
        Record power usage for a device synchronously.
        
        Args:
            device_id: Identifier for the device.
            watts: Power consumption in watts.
            duration_hours: Duration the device was active in hours.
            date_str: Optional date ISO string.
        """
        today = date_str or datetime.date.today().isoformat()
        if today not in self.daily_usage:
            self.daily_usage[today] = {}
        if today not in self.daily_cost:
            self.daily_cost[today] = {}

        kwh = (watts * duration_hours) / 1000.0
        cost = kwh * self.cost_per_kwh
        self.daily_usage[today][device_id] = self.daily_usage[today].get(device_id, 0.0) + kwh
        self.daily_cost[today][device_id] = self.daily_cost[today].get(device_id, 0.0) + cost

    def get_kwh(self, device_id: str, date_str: Optional[str] = None) -> float:
        """
        Get daily kWh accumulated for a device.
        """
        today = date_str or datetime.date.today().isoformat()
        return self.daily_usage.get(today, {}).get(device_id, 0.0)

    def get_cost(self, device_id: str, date_str: Optional[str] = None) -> float:
        """
        Get daily cost accumulated for a device.
        """
        today = date_str or datetime.date.today().isoformat()
        return self.daily_cost.get(today, {}).get(device_id, 0.0)

    def get_accumulated_cost(self, device_id: str, date_str: Optional[str] = None) -> float:
        """
        Alias for get_cost.
        """
        return self.get_cost(device_id, date_str)

    def get_daily_summary(self, date_str: Optional[str] = None) -> dict:
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
        cost_data = self.daily_cost.get(date_str, {})
        total_kwh = sum(day_data.values())
        if cost_data:
            total_cost = sum(cost_data.values())
        else:
            total_cost = total_kwh * self.cost_per_kwh

        return {
            "date": date_str,
            "device_usage_kwh": day_data,
            "total_kwh": round(total_kwh, 3),
            "estimated_cost_usd": round(total_cost, 2)
        }


analytics_engine = AnalyticsEngine()
