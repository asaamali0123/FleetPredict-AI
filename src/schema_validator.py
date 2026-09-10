from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


# These columns must exist in every future uploaded sensor CSV.
REQUIRED_COLUMNS = [
    "vehicle_id",
    "timestamp",
    "engine_speed_rpm",
    "vehicle_speed_kmh",
    "engine_load_pct",
    "accelerator_pedal_pct",
    "coolant_temp_c",
    "oil_pressure_kpa",
    "fuel_rail_pressure_mpa",
    "battery_voltage_v",
    "maf_g_s",
]

# These columns are useful when available, but not required.
OPTIONAL_COLUMNS = [
    "charging_status",
    "dtc_flag",
]

NUMERIC_REQUIRED_COLUMNS = [
    "engine_speed_rpm",
    "vehicle_speed_kmh",
    "engine_load_pct",
    "accelerator_pedal_pct",
    "coolant_temp_c",
    "oil_pressure_kpa",
    "fuel_rail_pressure_mpa",
    "battery_voltage_v",
    "maf_g_s",
]


def _base_result(row_count: int = 0, vehicle_count: int = 0) -> dict[str, Any]:
    """Create the standard validation-result structure."""
    return {
        "valid": False,
        "errors": [],
        "warnings": [],
        "row_count": row_count,
        "vehicle_count": vehicle_count,
    }


def _find_duplicate_columns(columns: list[str]) -> list[str]:
    """Return duplicate column names while keeping each name only once."""
    seen = set()
    duplicates = []

    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)

    return duplicates


def validate_sensor_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """
    Validate a sensor telemetry DataFrame against FleetPredict's schema.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Sensor telemetry data already loaded into Pandas.

    Returns
    -------
    dict
        Validation result containing valid, errors, warnings, row_count,
        and vehicle_count.
    """
    result = _base_result(row_count=len(dataframe))

    if dataframe.empty:
        result["errors"].append("The uploaded CSV has no data rows.")
        return result

    duplicate_columns = _find_duplicate_columns(dataframe.columns.tolist())
    if duplicate_columns:
        result["errors"].append(
            "Duplicate column name(s) found: " + ", ".join(duplicate_columns)
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    for column in missing_columns:
        result["errors"].append(f"Missing required column: {column}")

    # We cannot continue detailed checks if key columns do not exist.
    if missing_columns:
        return result

    vehicle_values = dataframe["vehicle_id"].fillna("").astype(str).str.strip()
    usable_vehicle_count = vehicle_values[vehicle_values != ""].nunique()
    result["vehicle_count"] = int(usable_vehicle_count)

    if usable_vehicle_count == 0:
        result["errors"].append(
            "Column 'vehicle_id' has no usable vehicle ID values."
        )

    timestamp_values = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
        format="mixed",
    )
    invalid_timestamp_count = int(timestamp_values.isna().sum())

    if invalid_timestamp_count > 0:
        result["errors"].append(
            f"Column 'timestamp' contains {invalid_timestamp_count} invalid or empty value(s)."
        )

    for column in NUMERIC_REQUIRED_COLUMNS:
        numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_numeric_count = int(numeric_values.isna().sum())

        if invalid_numeric_count > 0:
            result["errors"].append(
                f"Column '{column}' contains {invalid_numeric_count} non-numeric or empty value(s)."
            )

    # Optional DTC flag must be numeric if the user supplies it.
    if "dtc_flag" in dataframe.columns:
        dtc_values = pd.to_numeric(dataframe["dtc_flag"], errors="coerce")
        invalid_dtc_count = int(dtc_values.isna().sum())

        if invalid_dtc_count > 0:
            result["errors"].append(
                f"Optional column 'dtc_flag' contains {invalid_dtc_count} non-numeric or empty value(s)."
            )

    unexpected_columns = [
        column
        for column in dataframe.columns
        if column not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    ]
    if unexpected_columns:
        result["warnings"].append(
            "Additional column(s) were found and will be ignored by the core "
            "Member 1 pipeline: "
            + ", ".join(unexpected_columns)
        )

    missing_optional_columns = [
        column for column in OPTIONAL_COLUMNS if column not in dataframe.columns
    ]
    if missing_optional_columns:
        result["warnings"].append(
            "Optional column(s) not provided: " + ", ".join(missing_optional_columns)
        )

    result["valid"] = len(result["errors"]) == 0
    return result


def validate_sensor_csv(file_path: str | Path) -> dict[str, Any]:
    """
    Read and validate a sensor telemetry CSV file.

    Use this function when the user uploads a file path.
    """
    path = Path(file_path)
    result = _base_result()

    if not path.exists():
        result["errors"].append(f"CSV file was not found: {path}")
        return result

    if path.suffix.lower() != ".csv":
        result["errors"].append("Uploaded file must have a .csv extension.")
        return result

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), [])

        duplicate_columns = _find_duplicate_columns(header)

        dataframe = pd.read_csv(path)

    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as error:
        result["errors"].append(f"Could not read the uploaded CSV: {error}")
        return result

    result = validate_sensor_dataframe(dataframe)

    if duplicate_columns:
        duplicate_error = (
            "Duplicate column name(s) found in the CSV header: "
            + ", ".join(duplicate_columns)
        )

        if duplicate_error not in result["errors"]:
            result["errors"].append(duplicate_error)
            result["valid"] = False

    return result


def prepare_sensor_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean, typed, time-sorted DataFrame after successful validation.

    Raises
    ------
    ValueError
        If the DataFrame is invalid.
    """
    validation_result = validate_sensor_dataframe(dataframe)

    if not validation_result["valid"]:
        error_message = " | ".join(validation_result["errors"])
        raise ValueError(
            "Sensor data preparation stopped because validation failed: "
            f"{error_message}"
        )

    prepared_dataframe = dataframe.copy()

    prepared_dataframe["vehicle_id"] = (
        prepared_dataframe["vehicle_id"].astype(str).str.strip()
    )
    prepared_dataframe["timestamp"] = pd.to_datetime(
        prepared_dataframe["timestamp"],
        format="mixed",
    )

    for column in NUMERIC_REQUIRED_COLUMNS:
        prepared_dataframe[column] = pd.to_numeric(prepared_dataframe[column])

    if "dtc_flag" in prepared_dataframe.columns:
        prepared_dataframe["dtc_flag"] = pd.to_numeric(
            prepared_dataframe["dtc_flag"]
        )

    prepared_dataframe = prepared_dataframe.sort_values(
        by=["vehicle_id", "timestamp"]
    ).reset_index(drop=True)

    return prepared_dataframe