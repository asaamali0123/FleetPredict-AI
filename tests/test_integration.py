from pathlib import Path

import pandas as pd

from src.maintenance_service import load_maintenance_history
from src.ai_diagnosis import run_vehicle_diagnosis
from src.anomaly_detector import detect_vehicle_anomalies
from src.health_score import calculate_vehicle_health
from src.maintenance_service import (
    load_maintenance_history,
    build_maintenance_summary,
)
from src.rag_retriever import retrieve_context


SENSOR_CSV = Path(
    "data/sample/isuzu_6hk1_sensor_telemetry.csv"
)

MAINTENANCE_CSV = Path(
    "data/maintenance/isuzu_6hk1_maintenance_history.csv"
)


def test_vehicle_diagnosis_integration():
    vehicle_id = "ISZ-003"

    # Load sensor data
    sensor_df = pd.read_csv(SENSOR_CSV)

    assert not sensor_df.empty

    print("Sensor telemetry loaded successfully.")
    print(f"Sensor rows: {len(sensor_df)}")

    # Load maintenance history
    maintenance_df = load_maintenance_history(
        MAINTENANCE_CSV
    )

    assert not maintenance_df.empty

    print("Maintenance history loaded successfully.")
    print(f"Maintenance records: {len(maintenance_df)}")

    # Confirm selected vehicle exists
    assert vehicle_id in sensor_df["vehicle_id"].astype(str).unique()

    print(f"Testing vehicle: {vehicle_id}")

    # Run Member 1 anomaly detection
    anomaly_result = detect_vehicle_anomalies(
        sensor_df,
        vehicle_id,
    )

    assert anomaly_result["vehicle_id"] == vehicle_id
    assert "anomalies" in anomaly_result

    print("Member 1 anomaly detection completed.")
    print(f"Anomalies found: {anomaly_result['anomaly_count']}")

    # Run Member 1 health score
    health_result = calculate_vehicle_health(
        sensor_df,
        vehicle_id,
    )

    assert health_result["vehicle_id"] == vehicle_id
    assert "overall_health" in health_result
    assert "status" in health_result

    print("Member 1 health calculation completed.")
    print(f"Health score: {health_result['overall_health']}")
    print(f"Status: {health_result['status']}")

    # Run Member 3 maintenance history
    maintenance_summary = build_maintenance_summary(
        maintenance_df,
        vehicle_id,
    )

    assert maintenance_summary["vehicle_id"] == vehicle_id
    assert "total_records" in maintenance_summary
    assert "recent_repairs" in maintenance_summary
    assert "repeated_issues" in maintenance_summary

    print("Maintenance history analysis completed.")
    print(
        f"Vehicle maintenance records: "
        f"{maintenance_summary['total_records']}"
    )
    print(
        f"Repeated issues: "
        f"{maintenance_summary['repeated_issues']}"
    )

    # Run Member 2 RAG
    rag_result = retrieve_context(
        " ".join(
            anomaly.get("issue", "")
            for anomaly in anomaly_result["anomalies"]
            if anomaly.get("issue")
        )
    )

    assert "context" in rag_result
    assert "sources" in rag_result

    print("Member 2 RAG retrieval completed.")
    print(f"RAG sources found: {len(rag_result['sources'])}")
    print(f"RAG context preview: {rag_result['context'][:200]}")

    # Run complete FleetPredict AI diagnosis
    diagnosis_result = run_vehicle_diagnosis(
        sensor_df,
        maintenance_df,
        vehicle_id,
    )

    assert diagnosis_result["success"] is True
    assert diagnosis_result["vehicle_id"] == vehicle_id
    assert "diagnosis" in diagnosis_result

    print("Full AI diagnosis completed successfully.")
    print(f"Diagnosis: {diagnosis_result['diagnosis']}")