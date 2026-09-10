from pathlib import Path
from datetime import timedelta

import pandas as pd

from src.anomaly_detector import (
    detect_fleet_anomalies,
    detect_vehicle_anomalies,
)
from src.schema_validator import (
    prepare_sensor_dataframe,
    validate_sensor_csv,
    validate_sensor_dataframe,
)
from src.sensor_analytics import (
    get_available_vehicles,
    get_fleet_summary,
    get_latest_fleet_readings,
    get_latest_vehicle_reading,
    get_vehicle_records,
    get_vehicle_sensor_summary,
)
from src.health_score import (
    calculate_fleet_health,
    calculate_vehicle_health,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_CSV_PATH = (
    PROJECT_ROOT / "data" / "sample" / "isuzu_6hk1_sensor_telemetry.csv"
)


def test_valid_sensor_csv() -> pd.DataFrame:
    """Test that the prepared sensor CSV passes validation."""

    print("\n--- TEST 1: Valid sensor CSV ---")

    validation_result = validate_sensor_csv(VALID_CSV_PATH)
    print(validation_result)

    assert validation_result["valid"] is True
    assert validation_result["row_count"] > 0
    assert validation_result["vehicle_count"] > 0

    raw_dataframe = pd.read_csv(VALID_CSV_PATH)
    prepared_dataframe = prepare_sensor_dataframe(raw_dataframe)

    print("Valid CSV test passed.")
    return prepared_dataframe


def test_missing_required_column(valid_dataframe: pd.DataFrame) -> None:
    """Test that a missing required column stops validation."""

    print("\n--- TEST 2: Missing required column ---")

    invalid_dataframe = valid_dataframe.drop(columns=["oil_pressure_kpa"])
    validation_result = validate_sensor_dataframe(invalid_dataframe)

    print(validation_result)

    assert validation_result["valid"] is False
    assert "Missing required column: oil_pressure_kpa" in validation_result["errors"]

    print("Missing-column test passed.")


def test_invalid_timestamp() -> None:
    """Test that an invalid timestamp stops validation."""

    print("\n--- TEST 3: Invalid timestamp ---")

    raw_dataframe = pd.read_csv(VALID_CSV_PATH)
    invalid_dataframe = raw_dataframe.copy()
    invalid_dataframe.loc[0, "timestamp"] = "this-is-not-a-date"

    validation_result = validate_sensor_dataframe(invalid_dataframe)

    print(validation_result)

    assert validation_result["valid"] is False
    assert any(
        "Column 'timestamp' contains" in error
        for error in validation_result["errors"]
    )

    print("Invalid-timestamp test passed.")


def test_empty_dataframe() -> None:
    """Test that an empty DataFrame stops validation."""

    print("\n--- TEST 4: Empty DataFrame ---")

    empty_dataframe = pd.DataFrame()
    validation_result = validate_sensor_dataframe(empty_dataframe)

    print(validation_result)

    assert validation_result["valid"] is False
    assert "The uploaded CSV has no data rows." in validation_result["errors"]

    print("Empty-data test passed.")


def test_sensor_analytics(valid_dataframe: pd.DataFrame) -> None:
    """Test Member 1 analytics functions using valid sensor data."""

    print("\n--- TEST 5: Sensor analytics ---")

    vehicles = get_available_vehicles(valid_dataframe)
    selected_vehicle_id = vehicles[0]

    print(f"Available vehicles: {len(vehicles)}")
    print(f"Selected vehicle for test: {selected_vehicle_id}")

    vehicle_records = get_vehicle_records(valid_dataframe, selected_vehicle_id)
    latest_vehicle_reading = get_latest_vehicle_reading(
        valid_dataframe,
        selected_vehicle_id,
    )
    latest_fleet_readings = get_latest_fleet_readings(valid_dataframe)
    fleet_summary = get_fleet_summary(valid_dataframe)
    vehicle_summary = get_vehicle_sensor_summary(
        valid_dataframe,
        selected_vehicle_id,
    )

    assert len(vehicles) > 0
    assert len(vehicle_records) > 0
    assert latest_vehicle_reading["vehicle_id"] == selected_vehicle_id
    assert latest_vehicle_reading["timestamp"] == vehicle_records["timestamp"].max()
    assert len(latest_fleet_readings) == len(vehicles)
    assert latest_fleet_readings["vehicle_id"].is_unique
    assert fleet_summary["total_records"] == len(valid_dataframe)
    assert fleet_summary["total_vehicles"] == len(vehicles)
    assert vehicle_summary["vehicle_id"] == selected_vehicle_id
    assert vehicle_summary["record_count"] == len(vehicle_records)
    assert "coolant_temp_c" in vehicle_summary["sensor_statistics"]

    print("\nFleet summary:")
    print(fleet_summary)

    print("\nLatest selected vehicle reading:")
    print(latest_vehicle_reading)

    print("\nSelected vehicle coolant statistics:")
    print(vehicle_summary["sensor_statistics"]["coolant_temp_c"])

    print("Sensor analytics test passed.")


def make_selected_reading_latest(
    valid_dataframe: pd.DataFrame,
    selected_row: pd.Series,
) -> tuple[pd.DataFrame, str]:
    """
    Create valid one-vehicle test data where the selected real reading
    becomes that vehicle's latest reading.
    """

    selected_vehicle_id = str(selected_row["vehicle_id"])

    vehicle_data = valid_dataframe[
        valid_dataframe["vehicle_id"] == selected_vehicle_id
    ].copy()

    selected_row = selected_row.copy()
    selected_row["timestamp"] = (
    pd.Timestamp(vehicle_data["timestamp"].max()).to_pydatetime()
    + timedelta(minutes=1)
)

    scenario_dataframe = pd.concat(
        [vehicle_data, pd.DataFrame([selected_row])],
        ignore_index=True,
    )

    return scenario_dataframe, selected_vehicle_id


def get_required_scenario_row(
    dataframe: pd.DataFrame,
    condition: pd.Series,
    scenario_name: str,
) -> pd.Series:
    """Return one real row that matches an anomaly-test condition."""

    matching_rows = dataframe[condition]

    if matching_rows.empty:
        raise ValueError(
            f"No prepared-data row was found for scenario: {scenario_name}"
        )

    return matching_rows.iloc[0]


def test_anomaly_detector(valid_dataframe: pd.DataFrame) -> None:
    """Test healthy, warning, and critical scenarios from prepared data."""

    print("\n--- TEST 6: Anomaly detector ---")

    likely_idle = (
        (valid_dataframe["vehicle_speed_kmh"] <= 5)
        & (valid_dataframe["accelerator_pedal_pct"] <= 5)
        & (valid_dataframe["coolant_temp_c"] >= 75)
    )

    warmed_engine = valid_dataframe["coolant_temp_c"] >= 75
    near_1400_rpm = valid_dataframe["engine_speed_rpm"].between(1250, 1550)

    engine_idle_problem = likely_idle & (
        (valid_dataframe["engine_speed_rpm"] < 500)
        | (valid_dataframe["engine_speed_rpm"] > 550)
    )

    oil_problem = (
        warmed_engine
        & near_1400_rpm
        & (valid_dataframe["oil_pressure_kpa"] < 210)
    )

    fuel_problem = likely_idle & (
        valid_dataframe["fuel_rail_pressure_mpa"] < 25
    )

    healthy_condition = (
        (valid_dataframe["coolant_temp_c"] < 100)
        & (valid_dataframe["battery_voltage_v"] >= 24)
        & (~engine_idle_problem)
        & (~oil_problem)
        & (~fuel_problem)
        & (valid_dataframe["dtc_flag"] == 0)
    )

    healthy_row = get_required_scenario_row(
        valid_dataframe,
        healthy_condition,
        "healthy reading",
    )
    healthy_data, healthy_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        healthy_row,
    )
    healthy_result = detect_vehicle_anomalies(
        healthy_data,
        healthy_vehicle_id,
    )

    assert healthy_result["anomaly_count"] == 0
    print("Healthy-reading test passed.")

    oil_row = get_required_scenario_row(
        valid_dataframe,
        oil_problem,
        "low oil pressure",
    )
    oil_data, oil_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        oil_row,
    )
    oil_result = detect_vehicle_anomalies(oil_data, oil_vehicle_id)

    assert any(
        anomaly["issue"] == "Low Oil Pressure"
        and anomaly["severity"] == "Critical"
        for anomaly in oil_result["anomalies"]
    )
    print("Low-oil-pressure test passed.")

    fuel_row = get_required_scenario_row(
        valid_dataframe,
        fuel_problem,
        "low fuel rail pressure at idle",
    )
    fuel_data, fuel_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        fuel_row,
    )
    fuel_result = detect_vehicle_anomalies(fuel_data, fuel_vehicle_id)

    assert any(
        anomaly["issue"] == "Low Fuel Rail Pressure at Idle"
        and anomaly["severity"] == "Critical"
        for anomaly in fuel_result["anomalies"]
    )
    print("Low-fuel-rail-pressure test passed.")

    cooling_row = get_required_scenario_row(
        valid_dataframe,
        valid_dataframe["coolant_temp_c"] >= 110,
        "critical coolant temperature",
    )
    cooling_data, cooling_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        cooling_row,
    )
    cooling_result = detect_vehicle_anomalies(
        cooling_data,
        cooling_vehicle_id,
    )

    assert any(
        anomaly["issue"] == "Critically High Coolant Temperature"
        and anomaly["severity"] == "Critical"
        for anomaly in cooling_result["anomalies"]
    )
    print("Critical-cooling test passed.")

    electrical_row = get_required_scenario_row(
        valid_dataframe,
        valid_dataframe["battery_voltage_v"] < 22,
        "critical battery undercharge",
    )
    electrical_data, electrical_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        electrical_row,
    )
    electrical_result = detect_vehicle_anomalies(
        electrical_data,
        electrical_vehicle_id,
    )

    assert any(
        anomaly["issue"] == "Critical Battery Undercharge"
        and anomaly["severity"] == "Critical"
        for anomaly in electrical_result["anomalies"]
    )
    print("Critical-electrical test passed.")

    fleet_results = detect_fleet_anomalies(valid_dataframe)

    assert len(fleet_results) == valid_dataframe["vehicle_id"].nunique()
    assert all("anomalies" in result for result in fleet_results)

    print(f"Latest fleet results created: {len(fleet_results)}")
    print("Anomaly detector test passed.")
def test_health_score(valid_dataframe: pd.DataFrame) -> None:
    """Test Good, Warning, and Critical vehicle-health outcomes."""

    print("\n--- TEST 7: Health score ---")

    likely_idle = (
        (valid_dataframe["vehicle_speed_kmh"] <= 5)
        & (valid_dataframe["accelerator_pedal_pct"] <= 5)
        & (valid_dataframe["coolant_temp_c"] >= 75)
    )

    warmed_engine = valid_dataframe["coolant_temp_c"] >= 75
    near_1400_rpm = valid_dataframe["engine_speed_rpm"].between(1250, 1550)

    engine_idle_problem = likely_idle & (
        (valid_dataframe["engine_speed_rpm"] < 500)
        | (valid_dataframe["engine_speed_rpm"] > 550)
    )

    oil_problem = (
        warmed_engine
        & near_1400_rpm
        & (valid_dataframe["oil_pressure_kpa"] < 210)
    )

    fuel_problem = likely_idle & (
        valid_dataframe["fuel_rail_pressure_mpa"] < 25
    )

    healthy_condition = (
        (valid_dataframe["coolant_temp_c"] < 100)
        & (valid_dataframe["battery_voltage_v"] >= 24)
        & (~engine_idle_problem)
        & (~oil_problem)
        & (~fuel_problem)
        & (valid_dataframe["dtc_flag"] == 0)
    )

    healthy_row = get_required_scenario_row(
        valid_dataframe,
        healthy_condition,
        "healthy health-score reading",
    )
    healthy_data, healthy_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        healthy_row,
    )
    healthy_result = calculate_vehicle_health(
        healthy_data,
        healthy_vehicle_id,
    )

    assert healthy_result["status"] == "Good"
    assert healthy_result["overall_health"] == 100
    print("Healthy-score test passed.")

    warning_condition = valid_dataframe["coolant_temp_c"].between(
        100,
        109.999,
    )
    warning_row = get_required_scenario_row(
        valid_dataframe,
        warning_condition,
        "high coolant temperature warning",
    )
    warning_data, warning_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        warning_row,
    )
    warning_result = calculate_vehicle_health(
        warning_data,
        warning_vehicle_id,
    )

    assert warning_result["status"] == "Warning"
    assert warning_result["systems"]["cooling"] == 80
    assert warning_result["overall_health"] <= 80
    print("Warning-score test passed.")

    critical_row = get_required_scenario_row(
        valid_dataframe,
        oil_problem,
        "critical low oil pressure",
    )
    critical_data, critical_vehicle_id = make_selected_reading_latest(
        valid_dataframe,
        critical_row,
    )
    critical_result = calculate_vehicle_health(
        critical_data,
        critical_vehicle_id,
    )

    assert critical_result["status"] == "Critical"
    assert critical_result["systems"]["oil"] == 50
    assert critical_result["overall_health"] <= 50
    print("Critical-score test passed.")

    fleet_health_results = calculate_fleet_health(valid_dataframe)

    assert len(fleet_health_results) == valid_dataframe["vehicle_id"].nunique()
    assert all("overall_health" in result for result in fleet_health_results)

    print(f"Fleet health results created: {len(fleet_health_results)}")
    print("Health-score test passed.")
def test_complete_member1_pipeline() -> None:
    """Run the complete Member 1 pipeline on the uploaded sensor CSV."""

    print("\n--- TEST 8: Complete Member 1 pipeline ---")

    raw_dataframe = pd.read_csv(VALID_CSV_PATH)

    validation_result = validate_sensor_dataframe(raw_dataframe)
    assert validation_result["valid"] is True

    prepared_dataframe = prepare_sensor_dataframe(raw_dataframe)
    available_vehicles = get_available_vehicles(prepared_dataframe)
    latest_fleet_readings = get_latest_fleet_readings(prepared_dataframe)
    fleet_anomaly_results = detect_fleet_anomalies(prepared_dataframe)
    fleet_health_results = calculate_fleet_health(prepared_dataframe)

    selected_vehicle_id = available_vehicles[0]
    selected_latest_reading = get_latest_vehicle_reading(
        prepared_dataframe,
        selected_vehicle_id,
    )
    selected_anomaly_result = detect_vehicle_anomalies(
        prepared_dataframe,
        selected_vehicle_id,
    )
    selected_health_result = calculate_vehicle_health(
        prepared_dataframe,
        selected_vehicle_id,
    )

    assert len(available_vehicles) == validation_result["vehicle_count"]
    assert len(latest_fleet_readings) == len(available_vehicles)
    assert len(fleet_anomaly_results) == len(available_vehicles)
    assert len(fleet_health_results) == len(available_vehicles)
    assert selected_latest_reading["vehicle_id"] == selected_vehicle_id
    assert selected_anomaly_result["vehicle_id"] == selected_vehicle_id
    assert selected_health_result["vehicle_id"] == selected_vehicle_id

    critical_vehicles = [
        result
        for result in fleet_health_results
        if result["status"] == "Critical"
    ]
    warning_vehicles = [
        result
        for result in fleet_health_results
        if result["status"] == "Warning"
    ]
    good_vehicles = [
        result
        for result in fleet_health_results
        if result["status"] == "Good"
    ]

    print(f"Validated rows: {validation_result['row_count']}")
    print(f"Vehicles processed: {len(available_vehicles)}")
    print(f"Latest fleet readings: {len(latest_fleet_readings)}")
    print(f"Critical vehicles: {len(critical_vehicles)}")
    print(f"Warning vehicles: {len(warning_vehicles)}")
    print(f"Good vehicles: {len(good_vehicles)}")

    print("\nSelected vehicle pipeline result:")
    print(
        {
            "vehicle_id": selected_vehicle_id,
            "latest_timestamp": selected_latest_reading["timestamp"],
            "anomaly_count": selected_anomaly_result["anomaly_count"],
            "overall_health": selected_health_result["overall_health"],
            "status": selected_health_result["status"],
        }
    )

    print("Complete Member 1 pipeline test passed.")
def test_custom_vehicle_ids_and_schema() -> None:
    """Prove Member 1 code works with arbitrary vehicle IDs."""

    print("\n--- TEST 9: Custom vehicle IDs and required-only schema ---")

    custom_dataframe = pd.read_csv(VALID_CSV_PATH).head(12).copy()

    custom_dataframe["vehicle_id"] = [
        "FLEET-X-901" if index % 2 == 0 else "ROAD-TRUCK-Z"
        for index in range(len(custom_dataframe))
    ]

    columns_to_remove = [
        "charging_status",
        "dtc_flag",
        "oil_pressure_below_manual_check",
        "idle_out_of_target",
        "fuel_rail_low_at_idle",
        "cooling_high_demo_flag",
        "operating_scenario",
    ]

    custom_dataframe = custom_dataframe.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    # Give one custom vehicle a clear demo cooling anomaly.
    custom_dataframe.loc[
        custom_dataframe.index[-1],
        "coolant_temp_c",
    ] = 112.0

    validation_result = validate_sensor_dataframe(custom_dataframe)

    assert validation_result["valid"] is True
    assert validation_result["vehicle_count"] == 2

    custom_vehicles = get_available_vehicles(custom_dataframe)
    custom_anomalies = detect_fleet_anomalies(custom_dataframe)
    custom_health = calculate_fleet_health(custom_dataframe)

    assert set(custom_vehicles) == {"FLEET-X-901", "ROAD-TRUCK-Z"}
    assert len(custom_anomalies) == 2
    assert len(custom_health) == 2
    assert all(
        result["vehicle_id"] in custom_vehicles
        for result in custom_health
    )

    critical_custom_vehicle = next(
        result
        for result in custom_health
        if result["vehicle_id"] == "ROAD-TRUCK-Z"
    )

    assert critical_custom_vehicle["status"] == "Critical"
    assert critical_custom_vehicle["systems"]["cooling"] == 50

    print("Custom vehicle IDs:", custom_vehicles)
    print("Custom schema validation result:", validation_result)
    print("Custom critical vehicle result:", critical_custom_vehicle)
    print("Custom-ID and required-only schema test passed.")


if __name__ == "__main__":
    valid_dataframe = test_valid_sensor_csv()
    test_missing_required_column(valid_dataframe)
    test_invalid_timestamp()
    test_empty_dataframe()
    test_sensor_analytics(valid_dataframe)
    test_anomaly_detector(valid_dataframe)
    test_health_score(valid_dataframe)
    test_complete_member1_pipeline()
    test_custom_vehicle_ids_and_schema()

    print("\nAll current Member 1 tests passed.")