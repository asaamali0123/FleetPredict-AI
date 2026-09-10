from __future__ import annotations

from typing import Any

import pandas as pd

from src.anomaly_detector import (
    detect_fleet_anomalies,
    detect_vehicle_anomalies,
)


SYSTEM_NAMES = [
    "engine",
    "cooling",
    "oil",
    "fuel",
    "electrical",
]

# Every system begins at 100. These deductions use the approved MVP scoring plan.
ANOMALY_DEDUCTIONS = {
    "Idle Speed Out of Target Range": ("engine", 15),
    "High Coolant Temperature": ("cooling", 20),
    "Critically High Coolant Temperature": ("cooling", 50),
    "Low Oil Pressure": ("oil", 50),
    "Low Fuel Rail Pressure at Idle": ("fuel", 45),
    "Battery Undercharge": ("electrical", 20),
    "Critical Battery Undercharge": ("electrical", 50),
}



def calculate_health_from_anomalies(
    anomaly_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate one vehicle's health score from a detector result.

    Generic DTC alerts affect status but do not reduce a specific
    system score because they do not identify a specific system.
    """

    required_keys = ["vehicle_id", "timestamp", "anomalies"]

    for key in required_keys:
        if key not in anomaly_result:
            raise ValueError(
                f"Health scoring requires anomaly result key: '{key}'"
            )

    system_scores = {system: 100 for system in SYSTEM_NAMES}
    anomalies = anomaly_result["anomalies"]

    diagnostic_alerts = 0
    has_warning = False
    has_critical = False

    for anomaly in anomalies:
        issue = anomaly.get("issue", "")
        severity = anomaly.get("severity", "")
        system = anomaly.get("system", "")

        if severity == "Critical":
            has_critical = True
        elif severity == "Warning":
            has_warning = True

        if system == "Diagnostics":
            diagnostic_alerts += 1
            continue

        if issue in ANOMALY_DEDUCTIONS:
            score_system, deduction = ANOMALY_DEDUCTIONS[issue]
            system_scores[score_system] = max(
                0,
                system_scores[score_system] - deduction,
            )

    overall_health = min(system_scores.values())

    if has_critical:
        status = "Critical"
    elif has_warning or diagnostic_alerts > 0:
        status = "Warning"
    else:
        status = "Good"

    return {
        "vehicle_id": anomaly_result["vehicle_id"],
        "timestamp": anomaly_result["timestamp"],
        "overall_health": overall_health,
        "status": status,
        "systems": system_scores,
        "anomaly_count": len(anomalies),
        "diagnostic_alerts": diagnostic_alerts,
    }


def calculate_vehicle_health(
    dataframe: pd.DataFrame,
    vehicle_id: str,
) -> dict[str, Any]:
    """
    Calculate health from the latest sensor reading for one vehicle.

    Raises ValueError if the sensor data is invalid or the vehicle is absent.
    """

    anomaly_result = detect_vehicle_anomalies(dataframe, vehicle_id)

    return calculate_health_from_anomalies(anomaly_result)


def calculate_fleet_health(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Calculate latest health scores for every vehicle in a valid fleet CSV.
    """

    fleet_anomaly_results = detect_fleet_anomalies(dataframe)

    return [
        calculate_health_from_anomalies(anomaly_result)
        for anomaly_result in fleet_anomaly_results
    ]