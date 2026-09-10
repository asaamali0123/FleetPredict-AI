# FleetPredict AI — Member 1 Handoff

## Member 1 Status

Complete.

Member 1 modules provide:

- CSV schema validation
- Sensor-data preparation
- Fleet and vehicle analytics
- Rule-based anomaly detection
- Rule-based vehicle health scoring
- Tests for valid, invalid, healthy, warning, critical, and custom-schema cases

## Files Completed

```text
src/schema_validator.py
src/sensor_analytics.py
src/anomaly_detector.py
src/health_score.py
tests/test_member1.py
Public Functions
src/schema_validator.py
validate_sensor_csv(file_path)
validate_sensor_dataframe(dataframe)
prepare_sensor_dataframe(dataframe)

Use these functions first. If validation fails, do not continue to analytics, anomaly detection, health scoring, RAG, or AI diagnosis.

src/sensor_analytics.py
get_available_vehicles(dataframe)
get_vehicle_records(dataframe, vehicle_id)
get_latest_vehicle_reading(dataframe, vehicle_id)
get_latest_fleet_readings(dataframe)
get_fleet_summary(dataframe)
get_vehicle_sensor_summary(dataframe, vehicle_id)
src/anomaly_detector.py
detect_vehicle_anomalies(dataframe, vehicle_id)
detect_fleet_anomalies(dataframe)

Anomaly results use the latest reading for each vehicle.

src/health_score.py
calculate_health_from_anomalies(anomaly_result)
calculate_vehicle_health(dataframe, vehicle_id)
calculate_fleet_health(dataframe)
How to Import Member 1 Modules
import pandas as pd

from src.schema_validator import (
    prepare_sensor_dataframe,
    validate_sensor_dataframe,
)
from src.anomaly_detector import detect_fleet_anomalies
from src.health_score import calculate_fleet_health
Required Sensor CSV Schema

Every uploaded sensor CSV must contain these columns:

vehicle_id
timestamp
engine_speed_rpm
vehicle_speed_kmh
engine_load_pct
accelerator_pedal_pct
coolant_temp_c
oil_pressure_kpa
fuel_rail_pressure_mpa
battery_voltage_v
maf_g_s

Optional columns:

charging_status
dtc_flag

Users do not need to provide generated/demo columns such as:

oil_pressure_below_manual_check
idle_out_of_target
fuel_rail_low_at_idle
cooling_high_demo_flag
operating_scenario
Expected Input and Output

Input:

raw_dataframe = pd.read_csv(uploaded_csv_file)

Validation:

validation_result = validate_sensor_dataframe(raw_dataframe)

if not validation_result["valid"]:
    print(validation_result["errors"])
    # Stop further processing here.

Preparation:

prepared_dataframe = prepare_sensor_dataframe(raw_dataframe)

Fleet anomalies:

fleet_anomalies = detect_fleet_anomalies(prepared_dataframe)

Fleet health scores:

fleet_health = calculate_fleet_health(prepared_dataframe)

Example health result:

{
    "vehicle_id": "ANY-VEHICLE-ID",
    "timestamp": "2026-09-08T07:50:00",
    "overall_health": 50,
    "status": "Critical",
    "systems": {
        "engine": 100,
        "cooling": 50,
        "oil": 100,
        "fuel": 100,
        "electrical": 100
    },
    "anomaly_count": 1,
    "diagnostic_alerts": 0
}
Approved Anomaly Rules

Manual-grounded rules:

Idle speed outside 500–550 RPM when likely idling
Oil pressure below 210 kPa near 1,400 RPM after warm-up
Fuel rail pressure below 25 MPa when likely idling after warm-up

Demo rules:

Coolant temperature 100–109.9°C: Warning
Coolant temperature 110°C or above: Critical
Battery voltage 22–23.99 V: Warning
Battery voltage below 22 V: Critical

Every anomaly result includes rule_source so the application can clearly distinguish manual-grounded and demo rules.

Health Score Logic
Every system starts at 100.
Anomaly deductions are applied to the related system.
Overall health equals the lowest system score.
Any Critical anomaly gives Critical status.
Any Warning anomaly or DTC flag gives Warning status.
No anomalies gives Good status.
DTC alerts do not reduce a specific system score because a generic DTC does not identify one system.

This is a transparent hackathon MVP indicator, not an OEM-certified health model.

Dependencies

Current Member 1 dependency:

pandas>=2.2.0,<3.0.0

Future members should add their own packages to requirements.txt only when needed.

Test Command

Run from the project root:

python -m tests.test_member1

Expected final message:

All current Member 1 tests passed.
Known Limitations
Rules use the latest available reading, not long-term trend analysis.
No machine-learning anomaly model is used.
Cooling and electrical thresholds are clearly labelled demo rules.
charging_status is optional and does not control anomaly detection.
Generic DTC flags are alerts only. The future AI diagnosis module can explain them using the manual and maintenance history.
Important: Do Not Change

Members 2, 3, and 4 must not:

change the required schema column names
remove validation before processing
hard-code Isuzu vehicle IDs, row counts, or timestamps
use helper/demo input columns as the source of anomalies
relabel demo thresholds as official Isuzu rules
copy Member 1 logic into app.py

Member 4 should import and call these modules from app.py.


