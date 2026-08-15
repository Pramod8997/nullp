"""
EMS Pipeline Module
"""

from src.pipeline.safety import FleetDiagnosticsMonitor, SafetyMonitor
from src.pipeline.watchdog import Watchdog, WatchdogEvent, SoftAnomalyWatchdog
from src.pipeline.aggregate_nilm import NILMTransientDetector, OverlapAwareNILMDetector
from src.pipeline.delta_stability import DeltaStabilityAnalyzer
from src.pipeline.phantom_tracker import PhantomTracker
from src.pipeline.analytics import AnalyticsEngine, compute_tou_cost
from src.pipeline.calibration import compute_ece
from src.pipeline.classifier import ModeClassifier
from src.pipeline.failure_matrix import FailureMatrix
from src.pipeline.temporal_validator import TemporalValidator
from src.pipeline.stages import (
    NILMPreprocessor,
    ProtoNetClassifier,
    OpenMaxStage,
    TemperatureScalingStage,
    ConfidenceGateStage,
    DeltaStabilityStage,
    PhantomTrackerStage,
    AnalyticsStage,
    DigitalTwinStage,
    RLAgentStage,
    FullPipeline,
    BroadcastStage,
    mock_power_event,
    mock_power_event_with_window,
    mock_power_event_low_confidence,
    mock_thermal_event,
)

__all__ = [
    "FleetDiagnosticsMonitor",
    "SafetyMonitor",
    "Watchdog",
    "WatchdogEvent",
    "SoftAnomalyWatchdog",
    "NILMTransientDetector",
    "OverlapAwareNILMDetector",
    "DeltaStabilityAnalyzer",
    "PhantomTracker",
    "AnalyticsEngine",
    "compute_tou_cost",
    "compute_ece",
    "ModeClassifier",
    "FailureMatrix",
    "TemporalValidator",
    "NILMPreprocessor",
    "ProtoNetClassifier",
    "OpenMaxStage",
    "TemperatureScalingStage",
    "ConfidenceGateStage",
    "DeltaStabilityStage",
    "PhantomTrackerStage",
    "AnalyticsStage",
    "DigitalTwinStage",
    "RLAgentStage",
    "FullPipeline",
    "BroadcastStage",
    "mock_power_event",
    "mock_power_event_with_window",
    "mock_power_event_low_confidence",
    "mock_thermal_event",
]
