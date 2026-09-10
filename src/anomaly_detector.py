from __future__ import annotations

from typing import Any

import pandas as pd

from src.schema_validator import prepare_sensor_dataframe


def _add_anomaly(
    anomalies: list[dict[str, Any]],
    system: str,
    issue: str,
    severity: str,
    metric: str,
    value: float | int,
    rule_source: str,
) -> None:
    """Add one consistently structured anomaly to the anomaly list."""

    anomalies.append(
        {
            "system": system,
            "issue": issue,
            "severity": severity,
            "metric": metric,
            "value": round(float(value), 2),
            "rule_source": rule_source,
        }
    )


def _is_likely_idle(reading: pd.Series) -> bool:
    """
    Estimate whether a vehicle is idling using available telemetry.

    These conditions are a transparent demo implementation choice.
    """
    return bool(
        reading["vehicle_speed_kmh"] <= 5
        and reading["accelerator_pedal_pct"] <= 5
        and reading["coolant_temp_c"] >= 75
    )


def _detect_reading_anomalies(reading: pd.Series) -> list[dict[str, Any]]:
    """Detect approved anomalies in one already prepared sensor reading."""

    anomalies: list[dict[str, Any]] = []

    engine_speed = reading["engine_speed_rpm"]
    coolant_temp = reading["coolant_temp_c"]
    oil_pressure = reading["oil_pressure_kpa"]
    fuel_rail_pressure = reading["fuel_rail_pressure_mpa"]
    battery_voltage = reading["battery_voltage_v"]

    likely_idle = _is_likely_idle(reading)
    warmed_engine = coolant_temp >= 75
    near_1400_rpm = 1250 <= engine_speed <= 1550

    # Manual-grounded rule: specified idle speed is 500-550 RPM.
    if likely_idle and (engine_speed < 500 or engine_speed > 550):
        _add_anomaly(
            anomalies=anomalies,
            system="Engine",
            issue="Idle Speed Out of Target Range",
            severity="Warning",
            metric="engine_speed_rpm",
            value=engine_speed,
            rule_source="Manual-grounded rule",
        )

    # Manual-grounded threshold: oil pressure must exceed 210 kPa at 1400 RPM.
    if warmed_engine and near_1400_rpm and oil_pressure < 210:
        _add_anomaly(
            anomalies=anomalies,
            system="Oil",
            issue="Low Oil Pressure",
            severity="Critical",
            metric="oil_pressure_kpa",
            value=oil_pressure,
            rule_source="Manual-grounded rule",
        )

    # Manual-grounded threshold: warmed idle fuel rail pressure must exceed 25 MPa.
    if likely_idle and fuel_rail_pressure < 25:
        _add_anomaly(
            anomalies=anomalies,
            system="Fuel",
            issue="Low Fuel Rail Pressure at Idle",
            severity="Critical",
            metric="fuel_rail_pressure_mpa",
            value=fuel_rail_pressure,
            rule_source="Manual-grounded rule",
        )

    # Demo cooling rules.
    if coolant_temp >= 110:
        _add_anomaly(
            anomalies=anomalies,
            system="Cooling",
            issue="Critically High Coolant Temperature",
            severity="Critical",
            metric="coolant_temp_c",
            value=coolant_temp,
            rule_source="Demo rule",
        )
    elif coolant_temp >= 100:
        _add_anomaly(
            anomalies=anomalies,
            system="Cooling",
            issue="High Coolant Temperature",
            severity="Warning",
            metric="coolant_temp_c",
            value=coolant_temp,
            rule_source="Demo rule",
        )

    # Demo electrical rules for a 24 V system.
    if battery_voltage < 22:
        _add_anomaly(
            anomalies=anomalies,
            system="Electrical",
            issue="Critical Battery Undercharge",
            severity="Critical",
            metric="battery_voltage_v",
            value=battery_voltage,
            rule_source="Demo rule",
        )
    elif battery_voltage < 24:
        _add_anomaly(
            anomalies=anomalies,
            system="Electrical",
            issue="Battery Undercharge",
            severity="Warning",
            metric="battery_voltage_v",
            value=battery_voltage,
            rule_source="Demo rule",
        )

    # Optional DTC information from user-provided telemetry.
    if "dtc_flag" in reading.index and reading["dtc_flag"] > 0:
        _add_anomaly(
            anomalies=anomalies,
            system="Diagnostics",
            issue="Diagnostic Trouble Code Flag Reported",
            severity="Warning",
            metric="dtc_flag",
            value=reading["dtc_flag"],
            rule_source="Input-data flag",
        )

    return anomalies


def detect_vehicle_anomalies(
    dataframe: pd.DataFrame,
    vehicle_id: str,
) -> dict[str, Any]:
    """
    Detect anomalies in the latest reading for one vehicle.

    Raises ValueError if the uploaded data is invalid or vehicle_id is absent.
    """

    prepared_dataframe = prepare_sensor_dataframe(dataframe)
    cleaned_vehicle_id = str(vehicle_id).strip()

    vehicle_records = prepared_dataframe[
        prepared_dataframe["vehicle_id"] == cleaned_vehicle_id
    ]

    if vehicle_records.empty:
        raise ValueError(
            f"Vehicle ID '{cleaned_vehicle_id}' was not found in the sensor data."
        )

    latest_reading = vehicle_records.iloc[-1]
    anomalies = _detect_reading_anomalies(latest_reading)

    return {
        "vehicle_id": cleaned_vehicle_id,
        "timestamp": latest_reading["timestamp"].isoformat(),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def detect_fleet_anomalies(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect anomalies in the latest reading of every available vehicle.

    Returns one structured result per vehicle, including healthy vehicles
    with an empty anomaly list.
    """

    prepared_dataframe = prepare_sensor_dataframe(dataframe)

    latest_readings = (
        prepared_dataframe.groupby("vehicle_id", as_index=False)
        .tail(1)
        .sort_values("vehicle_id")
    )

    fleet_results = []

    for _, reading in latest_readings.iterrows():
        anomalies = _detect_reading_anomalies(reading)

        fleet_results.append(
            {
                "vehicle_id": reading["vehicle_id"],
                "timestamp": reading["timestamp"].isoformat(),
                "anomaly_count": len(anomalies),
                "anomalies": anomalies,
            }
        )

    return fleet_results