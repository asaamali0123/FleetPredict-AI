import pandas as pd


REQUIRED_COLUMNS = [
    "maintenance_id",
    "vehicle_id",
    "service_date",
    "odometer_km",
    "system",
    "detected_issue",
    "probable_cause",
    "repair_action",
    "component",
    "manual_reference",
    "maintenance_type",
    "downtime_hours",
    "status",
]


def load_maintenance_history(csv_path):
    """
    Load and validate a maintenance history CSV file.

    Returns:
        pandas.DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Maintenance CSV not found: {csv_path}"
        )
    except Exception as exc:
        raise ValueError(
            f"Could not read maintenance CSV: {exc}"
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required maintenance columns: {missing_columns}"
        )

    # Convert service dates into proper datetime values.
    df["service_date"] = pd.to_datetime(
        df["service_date"],
        errors="coerce"
    )

    return df


def get_vehicle_history(df, vehicle_id):
    """
    Return all maintenance records for a vehicle.

    Records are sorted from newest to oldest.
    """
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()

    vehicle_history = df[
        df["vehicle_id"].astype(str) == str(vehicle_id)
    ].copy()

    vehicle_history = vehicle_history.sort_values(
        by="service_date",
        ascending=False
    )

    return vehicle_history.reset_index(drop=True)


def get_recent_maintenance(df, vehicle_id, limit=5):
    """
    Return the most recent maintenance records for a vehicle.
    """
    history = get_vehicle_history(df, vehicle_id)

    if history.empty:
        return []

    recent = history.head(limit)

    records = []

    for _, row in recent.iterrows():
        records.append({
            "maintenance_id": row["maintenance_id"],
            "service_date": (
                row["service_date"].strftime("%Y-%m-%d")
                if pd.notna(row["service_date"])
                else None
            ),
            "system": row["system"],
            "detected_issue": row["detected_issue"],
            "probable_cause": row["probable_cause"],
            "repair_action": row["repair_action"],
            "component": row["component"],
            "maintenance_type": row["maintenance_type"],
            "status": row["status"],
            "downtime_hours": row["downtime_hours"],
        })

    return records


def find_repeated_issues(df, vehicle_id):
    """
    Find systems and detected issues that appear more than once
    in a vehicle's maintenance history.
    """
    history = get_vehicle_history(df, vehicle_id)

    if history.empty:
        return {
            "repeated_systems": [],
            "repeated_issues": [],
        }

    system_counts = (
        history["system"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    issue_counts = (
        history["detected_issue"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    repeated_systems = [
        {
            "system": system,
            "count": int(count),
        }
        for system, count in system_counts.items()
        if count > 1
    ]

    repeated_issues = [
        {
            "issue": issue,
            "count": int(count),
        }
        for issue, count in issue_counts.items()
        if count > 1
    ]

    return {
        "repeated_systems": repeated_systems,
        "repeated_issues": repeated_issues,
    }


def build_maintenance_summary(df, vehicle_id):
    """
    Build a simple maintenance summary for one vehicle.
    """
    history = get_vehicle_history(df, vehicle_id)

    repeated = find_repeated_issues(df, vehicle_id)

    if history.empty:
        return {
            "vehicle_id": str(vehicle_id),
            "total_records": 0,
            "latest_service_date": None,
            "recent_repairs": [],
            "repeated_systems": [],
            "repeated_issues": [],
        }

    latest_date = history["service_date"].iloc[0]

    latest_service_date = (
        latest_date.strftime("%Y-%m-%d")
        if pd.notna(latest_date)
        else None
    )

    return {
        "vehicle_id": str(vehicle_id),
        "total_records": int(len(history)),
        "latest_service_date": latest_service_date,
        "recent_repairs": get_recent_maintenance(
            df,
            vehicle_id,
            limit=5
        ),
        "repeated_systems": repeated["repeated_systems"],
        "repeated_issues": repeated["repeated_issues"],
    }