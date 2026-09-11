from pathlib import Path

import pandas as pd
import pytest

from src.maintenance_service import (
    REQUIRED_COLUMNS,
    load_maintenance_history,
    get_vehicle_history,
    get_recent_maintenance,
    find_repeated_issues,
    build_maintenance_summary,
)


CSV_PATH = Path("data/maintenance/isuzu_6hk1_maintenance_history.csv")


def test_csv_loads():
    df = load_maintenance_history(CSV_PATH)

    assert not df.empty
    assert len(df) == 96


def test_required_schema():
    df = load_maintenance_history(CSV_PATH)

    for column in REQUIRED_COLUMNS:
        assert column in df.columns


def test_service_date_is_parsed():
    df = load_maintenance_history(CSV_PATH)

    assert pd.api.types.is_datetime64_any_dtype(
        df["service_date"]
    )


def test_vehicle_filtering():
    df = load_maintenance_history(CSV_PATH)

    history = get_vehicle_history(df, "ISZ-001")

    assert not history.empty
    assert all(history["vehicle_id"] == "ISZ-001")


def test_unknown_vehicle():
    df = load_maintenance_history(CSV_PATH)

    history = get_vehicle_history(df, "UNKNOWN-999")

    assert history.empty


def test_recent_maintenance():
    df = load_maintenance_history(CSV_PATH)

    recent = get_recent_maintenance(
        df,
        "ISZ-001",
        limit=5,
    )

    assert len(recent) <= 5
    assert len(recent) > 0

    dates = [record["service_date"] for record in recent]

    assert dates == sorted(dates, reverse=True)


def test_repeated_issues():
    df = load_maintenance_history(CSV_PATH)

    result = find_repeated_issues(
        df,
        "ISZ-001",
    )

    assert "repeated_systems" in result
    assert "repeated_issues" in result

    assert len(result["repeated_systems"]) > 0
    assert len(result["repeated_issues"]) > 0


def test_unknown_vehicle_repeated_issues():
    df = load_maintenance_history(CSV_PATH)

    result = find_repeated_issues(
        df,
        "UNKNOWN-999",
    )

    assert result == {
        "repeated_systems": [],
        "repeated_issues": [],
    }


def test_maintenance_summary():
    df = load_maintenance_history(CSV_PATH)

    summary = build_maintenance_summary(
        df,
        "ISZ-001",
    )

    assert summary["vehicle_id"] == "ISZ-001"
    assert summary["total_records"] == 2
    assert summary["latest_service_date"] == "2026-08-15"

    assert isinstance(summary["recent_repairs"], list)
    assert isinstance(summary["repeated_systems"], list)
    assert isinstance(summary["repeated_issues"], list)


def test_unknown_vehicle_summary():
    df = load_maintenance_history(CSV_PATH)

    summary = build_maintenance_summary(
        df,
        "UNKNOWN-999",
    )

    assert summary["vehicle_id"] == "UNKNOWN-999"
    assert summary["total_records"] == 0
    assert summary["latest_service_date"] is None
    assert summary["recent_repairs"] == []
    assert summary["repeated_systems"] == []
    assert summary["repeated_issues"] == []