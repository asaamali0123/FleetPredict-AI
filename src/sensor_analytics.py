from __future__ import annotations

from typing import Any

import pandas as pd

from src.schema_validator import (
    NUMERIC_REQUIRED_COLUMNS,
    prepare_sensor_dataframe,
)


def _get_prepared_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare sensor data before any analytics are performed.

    This function stops processing with ValueError if validation fails.
    """
    return prepare_sensor_dataframe(dataframe)


def get_available_vehicles(dataframe: pd.DataFrame) -> list[str]:
    """Return all available vehicle IDs in alphabetical order."""

    prepared_dataframe = _get_prepared_dataframe(dataframe)

    return sorted(prepared_dataframe["vehicle_id"].unique().tolist())


def get_vehicle_records(
    dataframe: pd.DataFrame,
    vehicle_id: str,
) -> pd.DataFrame:
    """Return all time-sorted records for one vehicle."""

    prepared_dataframe = _get_prepared_dataframe(dataframe)
    cleaned_vehicle_id = str(vehicle_id).strip()

    vehicle_records = prepared_dataframe[
        prepared_dataframe["vehicle_id"] == cleaned_vehicle_id
    ].copy()

    if vehicle_records.empty:
        raise ValueError(
            f"Vehicle ID '{cleaned_vehicle_id}' was not found in the sensor data."
        )

    return vehicle_records.reset_index(drop=True)


def get_latest_vehicle_reading(
    dataframe: pd.DataFrame,
    vehicle_id: str,
) -> dict[str, Any]:
    """Return the latest sensor reading for one selected vehicle."""

    vehicle_records = get_vehicle_records(dataframe, vehicle_id)
    latest_record = vehicle_records.iloc[-1]

    return latest_record.to_dict()


def get_latest_fleet_readings(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the latest available reading for every vehicle."""

    prepared_dataframe = _get_prepared_dataframe(dataframe)

    latest_readings = (
        prepared_dataframe.groupby("vehicle_id", as_index=False)
        .tail(1)
        .sort_values("vehicle_id")
        .reset_index(drop=True)
    )

    return latest_readings


def get_fleet_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a simple fleet-level summary for the dashboard."""

    prepared_dataframe = _get_prepared_dataframe(dataframe)

    summary: dict[str, Any] = {
        "total_records": int(len(prepared_dataframe)),
        "total_vehicles": int(prepared_dataframe["vehicle_id"].nunique()),
        "first_timestamp": prepared_dataframe["timestamp"].min(),
        "last_timestamp": prepared_dataframe["timestamp"].max(),
        "average_engine_speed_rpm": round(
            float(prepared_dataframe["engine_speed_rpm"].mean()), 2
        ),
        "average_vehicle_speed_kmh": round(
            float(prepared_dataframe["vehicle_speed_kmh"].mean()), 2
        ),
        "average_coolant_temp_c": round(
            float(prepared_dataframe["coolant_temp_c"].mean()), 2
        ),
        "average_battery_voltage_v": round(
            float(prepared_dataframe["battery_voltage_v"].mean()), 2
        ),
    }

    if "dtc_flag" in prepared_dataframe.columns:
        summary["dtc_flagged_records"] = int(
            (prepared_dataframe["dtc_flag"] > 0).sum()
        )

    return summary


def get_vehicle_sensor_summary(
    dataframe: pd.DataFrame,
    vehicle_id: str,
) -> dict[str, Any]:
    """
    Return useful sensor averages, minimums, maximums,
    period, and latest timestamp for one vehicle.
    """

    vehicle_records = get_vehicle_records(dataframe, vehicle_id)

    sensor_summary: dict[str, Any] = {
        "vehicle_id": str(vehicle_id).strip(),
        "record_count": int(len(vehicle_records)),
        "first_timestamp": vehicle_records["timestamp"].min(),
        "last_timestamp": vehicle_records["timestamp"].max(),
        "sensor_statistics": {},
    }

    for column in NUMERIC_REQUIRED_COLUMNS:
        sensor_summary["sensor_statistics"][column] = {
            "average": round(float(vehicle_records[column].mean()), 2),
            "minimum": round(float(vehicle_records[column].min()), 2),
            "maximum": round(float(vehicle_records[column].max()), 2),
        }

    return sensor_summary