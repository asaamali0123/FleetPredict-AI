import os

from io import BytesIO



import pandas as pd

import plotly.express as px

import streamlit as st



from src.ai_diagnosis import run_vehicle_diagnosis

from src.anomaly_detector import detect_fleet_anomalies, detect_vehicle_anomalies

from src.health_score import calculate_fleet_health, calculate_vehicle_health

from src.maintenance_service import (

    REQUIRED_COLUMNS as MAINTENANCE_REQUIRED_COLUMNS,

    build_maintenance_summary,

    get_vehicle_history,

)

from src.schema_validator import prepare_sensor_dataframe, validate_sensor_dataframe

from src.sensor_analytics import (

    get_available_vehicles,

    get_latest_vehicle_reading,

    get_vehicle_records,

)



# =========================================================

# PAGE

# =========================================================

st.set_page_config(

    page_title="FleetPredict AI",

    page_icon="🚛",

    layout="wide",

    initial_sidebar_state="expanded",

)



# =========================================================

# THEME

# =========================================================

C = {

    "bg": "#0B1220",

    "panel": "#111B2E",

    "panel2": "#17243A",

    "panel3": "#1D2E49",

    "border": "#2C3E5E",

    "text": "#F3F7FC",

    "muted": "#9FB0C8",

    "blue": "#4DA3FF",

    "cyan": "#55D6E8",

    "teal": "#35C59A",

    "amber": "#F3C45A",

    "red": "#F47B64",

    "purple": "#B88CFF",

}



st.markdown(

    f"""

    <style>

    :root {{

      --bg:{C["bg"]}; --panel:{C["panel"]}; --panel2:{C["panel2"]};

      --panel3:{C["panel3"]}; --border:{C["border"]}; --text:{C["text"]};

      --muted:{C["muted"]}; --blue:{C["blue"]}; --cyan:{C["cyan"]};

      --teal:{C["teal"]}; --amber:{C["amber"]}; --red:{C["red"]};

    }}



    .stApp {{

      background:

        radial-gradient(circle at 12% -10%, rgba(77,163,255,.13), transparent 26%),

        radial-gradient(circle at 92% 0%, rgba(53,197,154,.09), transparent 22%),

        var(--bg);

      color: var(--text);

    }}



    .block-container {{

      max-width: 1480px;

      padding: .85rem 1.8rem 2rem 1.8rem;

    }}



    h1,h2,h3,h4,p,span,label {{ color: var(--text); }}



    /* Sidebar */

    section[data-testid="stSidebar"] {{

      background: linear-gradient(180deg, #0F1829 0%, #0B1321 100%);

      border-right: 1px solid var(--border);

    }}

    section[data-testid="stSidebar"] > div:first-child {{

      padding-top: .7rem;

    }}



    /* Brand */

    .brand {{

      display:flex; align-items:center; gap:10px; margin:0 0 4px 0;

    }}

    .brand-icon {{

      width:40px; height:40px; border-radius:12px;

      display:grid; place-items:center; font-size:1.42rem;

      background:linear-gradient(135deg,#2E7FD3,#2CAD86);

      box-shadow:0 8px 22px rgba(46,127,211,.25);

    }}

    .brand-title {{font-size:1.08rem; font-weight:850; line-height:1.1;}}

    .brand-sub {{font-size:.72rem; color:var(--muted); margin-top:2px;}}

    .side-label {{

      color:var(--muted); font-size:.68rem; font-weight:850;

      letter-spacing:.09em; text-transform:uppercase; margin:10px 0 6px;

    }}



    /* Navigation radio -> compact nav cards */

    section[data-testid="stSidebar"] div[role="radiogroup"] {{

      gap:5px;

    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label {{

      background:rgba(255,255,255,.025);

      border:1px solid transparent;

      border-radius:10px;

      padding:7px 9px;

      margin:0;

      transition:.15s ease;

    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{

      background:rgba(77,163,255,.08);

      border-color:rgba(77,163,255,.2);

    }}



    /* Expander */

    div[data-testid="stExpander"] {{

      border:1px solid var(--border);

      border-radius:13px;

      background:rgba(255,255,255,.02);

    }}



    /* Uploaders */

    [data-testid="stFileUploader"] {{

      border-radius:12px;

    }}

    [data-testid="stFileUploaderDropzone"] {{

      min-height:76px !important;

      padding:10px !important;

      background:var(--panel2) !important;

      border:1px dashed #466083 !important;

      border-radius:12px !important;

    }}

    [data-testid="stFileUploaderDropzone"] button {{

      background:#223554 !important;

      color:white !important;

      border:1px solid #3D5A82 !important;

      border-radius:9px !important;

    }}



    /* Main hero */

    .hero {{

      position:relative;

      overflow:hidden;

      border:1px solid rgba(255,255,255,.08);

      background:linear-gradient(115deg,#256FB5 0%,#277F91 52%,#258A71 100%);

      border-radius:20px;

      padding:22px 24px;

      margin-bottom:18px;

      box-shadow:0 16px 42px rgba(0,0,0,.23);

    }}

    .hero:after {{

      content:"";

      position:absolute; right:-50px; top:-70px;

      width:220px; height:220px; border-radius:50%;

      background:rgba(255,255,255,.06);

    }}

    .hero-kicker {{

      color:#CBEAFF; font-size:.72rem; font-weight:850;

      letter-spacing:.09em; text-transform:uppercase;

      margin-bottom:5px;

    }}

    .hero h1 {{

      margin:0; color:white; font-size:clamp(1.7rem,2.8vw,2.35rem);

      letter-spacing:-.03em; line-height:1.05;

    }}

    .hero p {{

      color:rgba(255,255,255,.88); margin:8px 0 0;

      max-width:900px; font-size:.96rem;

    }}



    /* Cards + metrics */

    .card {{

      background:linear-gradient(180deg,var(--panel2),var(--panel));

      border:1px solid var(--border);

      border-radius:16px;

      padding:16px 17px;

      box-shadow:0 9px 26px rgba(0,0,0,.17);

      margin-bottom:10px;

    }}

    .card-title {{font-weight:800; margin-bottom:5px;}}

    .muted {{color:var(--muted);}}

    .tiny {{font-size:.82rem;}}



    div[data-testid="stMetric"] {{

      background:linear-gradient(180deg,var(--panel2),var(--panel));

      border:1px solid var(--border);

      border-radius:16px;

      padding:15px 16px;

      min-height:102px;

      box-shadow:0 8px 24px rgba(0,0,0,.16);

    }}

    div[data-testid="stMetricLabel"] {{color:var(--muted)!important;font-weight:700;}}

    div[data-testid="stMetricValue"] {{color:var(--text)!important;font-weight:850;}}



    /* Status pills */

    .pill {{

      display:inline-flex; align-items:center; gap:7px;

      padding:7px 11px; border-radius:999px; font-weight:800; font-size:.82rem;

      border:1px solid transparent;

    }}

    .good {{background:rgba(53,197,154,.13);color:#A3F0D6;border-color:rgba(53,197,154,.32);}}

    .warn {{background:rgba(243,196,90,.13);color:#FFE29A;border-color:rgba(243,196,90,.32);}}

    .crit {{background:rgba(244,123,100,.13);color:#FFC0B4;border-color:rgba(244,123,100,.32);}}



    /* Dataframes and controls */

    div[data-testid="stDataFrame"] {{

      border:1px solid var(--border);

      border-radius:14px;

      overflow:hidden;

      background:var(--panel);

    }}

    div[data-baseweb="select"] > div {{

      background:var(--panel2)!important;

      border-color:var(--border)!important;

      border-radius:11px!important;

      color:var(--text)!important;

    }}

    .stButton > button {{

      border-radius:11px;

      min-height:44px;

      font-weight:800;

      color:white;

      border:1px solid #4A7FAD;

      background:linear-gradient(135deg,#2E78C1,#2A9A7C);

      box-shadow:0 8px 22px rgba(46,120,193,.18);

    }}



    /* Schema chips - replaces bright code box */

    .schema-grid {{

      display:grid;

      grid-template-columns:repeat(2,minmax(0,1fr));

      gap:8px;

    }}

    .schema-chip {{

      background:#15243A;

      border:1px solid #2D4262;

      color:#DCE9FA;

      border-radius:10px;

      padding:8px 10px;

      font-family:ui-monospace,SFMono-Regular,Menlo,monospace;

      font-size:.78rem;

      overflow-wrap:anywhere;

    }}



    .evidence {{

      background:rgba(255,255,255,.025);

      border:1px solid var(--border);

      border-left:4px solid var(--blue);

      border-radius:12px;

      padding:13px 14px;

      margin-bottom:9px;

    }}



    .footer {{

      text-align:center; color:var(--muted); font-size:.72rem; padding-top:18px;

    }}



    .stAlert {{border-radius:12px;border:1px solid var(--border);}}



    @media(max-width:950px){{

      .block-container{{padding-left:1rem;padding-right:1rem;}}

      .hero{{padding:18px 19px;}}

    }}



    @media(max-width:650px){{

      .block-container{{padding:.8rem .7rem 1.5rem;}}

      .hero{{border-radius:16px;padding:17px 16px;}}

      .hero h1{{font-size:1.65rem;}}

      .hero p{{font-size:.9rem;}}

      .schema-grid{{grid-template-columns:1fr;}}

      div[data-testid="stMetric"]{{min-height:92px;padding:12px;}}

    }}



    /* =====================================================

       FINAL UI POLISH — HEADER, SIDEBAR, EXPANDER, FONTS

       ===================================================== */



    header[data-testid="stHeader"],

    [data-testid="stHeader"],

    .stAppHeader,

    [data-testid="stToolbar"],

    .stAppToolbar {{

      background: #0B1220 !important;

      color: var(--text) !important;

      border-bottom: 1px solid rgba(44,62,94,.45) !important;

    }}



    [data-testid="stDecoration"] {{

      background: transparent !important;

      display: none !important;

    }}



    header[data-testid="stHeader"] button,

    header[data-testid="stHeader"] svg,

    [data-testid="stToolbar"] button,

    [data-testid="stToolbar"] svg {{

      color: #DDE8F7 !important;

      fill: #DDE8F7 !important;

    }}



    html, body, .stApp,

    button, input, textarea, select,

    [data-testid="stMarkdownContainer"],

    [data-testid="stWidgetLabel"],

    div[data-baseweb="select"] {{

      font-family: "Segoe UI", Arial, sans-serif !important;

    }}



    body, .stApp {{

      font-size: 15px !important;

    }}



    h1 {{ font-size: clamp(2rem, 3vw, 2.7rem) !important; font-weight: 800 !important; }}

    h2 {{ font-size: clamp(1.45rem, 2.1vw, 2rem) !important; font-weight: 780 !important; }}

    h3 {{ font-size: 1.2rem !important; font-weight: 760 !important; }}



    .brand {{

      gap: 12px !important;

      margin: 2px 0 10px 0 !important;

    }}



    .brand-icon {{

      width: 46px !important;

      height: 46px !important;

      border-radius: 13px !important;

      font-size: 1.5rem !important;

      flex: 0 0 46px !important;

    }}



    .brand-title {{

      color: #F5F8FD !important;

      font-size: 1.28rem !important;

      font-weight: 850 !important;

      letter-spacing: -.02em !important;

      line-height: 1.12 !important;

    }}



    .brand-sub {{

      color: #80CFFF !important;

      font-size: .79rem !important;

      font-weight: 600 !important;

      margin-top: 4px !important;

    }}



    .side-label {{

      color: #8FC8F2 !important;

      font-size: .72rem !important;

      font-weight: 850 !important;

      letter-spacing: .11em !important;

      margin: 14px 0 7px !important;

    }}



    section[data-testid="stSidebar"] div[role="radiogroup"] label {{

      padding: 9px 10px !important;

      border-radius: 11px !important;

      min-height: 38px !important;

    }}



    section[data-testid="stSidebar"] div[role="radiogroup"] label p,

    section[data-testid="stSidebar"] div[role="radiogroup"] label span {{

      font-size: .88rem !important;

      font-weight: 700 !important;

      color: #F0F5FC !important;

    }}



    details[data-testid="stExpander"],

    div[data-testid="stExpander"],

    [data-testid="stExpander"] details {{

      background: #101B2D !important;

      border: 1px solid #2C3E5E !important;

      border-radius: 13px !important;

      overflow: hidden !important;

    }}



    details[data-testid="stExpander"] > summary,

    [data-testid="stExpander"] summary,

    div[data-testid="stExpander"] summary {{

      background: #15243A !important;

      color: #F3F7FC !important;

      border: 0 !important;

      border-bottom: 1px solid #2C3E5E !important;

      padding: 10px 12px !important;

      min-height: 42px !important;

    }}



    details[data-testid="stExpander"] > summary:hover,

    [data-testid="stExpander"] summary:hover {{

      background: #1A2C46 !important;

    }}



    details[data-testid="stExpander"] > summary p,

    [data-testid="stExpander"] summary p,

    [data-testid="stExpander"] summary span,

    [data-testid="stExpander"] summary svg {{

      color: #F3F7FC !important;

      fill: #F3F7FC !important;

      font-weight: 750 !important;

      font-size: .86rem !important;

    }}



    [data-testid="stExpanderDetails"] {{

      background: #101B2D !important;

      padding: 10px 11px 12px !important;

    }}



    [data-testid="stFileUploader"] label p {{

      color: #EDF4FC !important;

      font-size: .84rem !important;

      font-weight: 700 !important;

    }}



    [data-testid="stFileUploaderDropzone"] small,

    [data-testid="stFileUploaderDropzoneInstructions"] span {{

      color: #9FB0C8 !important;

    }}



    .block-container {{

      padding-top: 1rem !important;

      padding-bottom: 1.6rem !important;

    }}



    .hero {{

      margin-top: 0 !important;

    }}



    .schema-chip {{

      font-size: .82rem !important;

      font-weight: 650 !important;

      padding: 9px 11px !important;

      background: #172740 !important;

    }}



    .card,

    .card p,

    [data-testid="stMarkdownContainer"] p,

    [data-testid="stMarkdownContainer"] li {{

      font-size: .94rem !important;

      line-height: 1.55 !important;

    }}



    @media(max-width:650px) {{

      .brand-title {{ font-size: 1.1rem !important; }}

      .brand-sub {{ font-size: .72rem !important; }}

      .brand-icon {{ width: 40px !important; height: 40px !important; flex-basis: 40px !important; }}

      section[data-testid="stSidebar"] div[role="radiogroup"] label p,

      section[data-testid="stSidebar"] div[role="radiogroup"] label span {{

        font-size: .83rem !important;

      }}

    }}





    /* FINAL FIX: remove Streamlit top bar completely */

    header[data-testid="stHeader"],

    [data-testid="stHeader"],

    .stAppHeader,

    [data-testid="stToolbar"],

    .stAppToolbar,

    [data-testid="stDecoration"] {{

        display: none !important;

        visibility: hidden !important;

        height: 0 !important;

        min-height: 0 !important;

    }}



    /* Remove space reserved for the hidden header */

    [data-testid="stAppViewContainer"],

    [data-testid="stAppViewContainer"] > .main,

    section.main {{

        padding-top: 0 !important;

        margin-top: 0 !important;

        background: #0B1220 !important;

    }}



    .block-container {{

        padding-top: 0.75rem !important;

        margin-top: 0 !important;

    }}



    /* Sidebar starts from very top too */

    section[data-testid="stSidebar"] {{

        top: 0 !important;

        height: 100vh !important;

    }}



    
    /* MOBILE FIX: restore sidebar access on phones */
    @media (max-width: 768px) {{
        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        .stAppHeader {{
            display: flex !important;
            visibility: visible !important;
            height: 3rem !important;
            min-height: 3rem !important;
            background: #0B1220 !important;
            border-bottom: 1px solid rgba(44,62,94,.45) !important;
        }}

        [data-testid="stToolbar"],
        .stAppToolbar,
        [data-testid="stDecoration"] {{
            display: none !important;
        }}

        [data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapsedControl"] {{
            display: flex !important;
            visibility: visible !important;
        }}

        .block-container {{
            padding-top: .6rem !important;
        }}
    }}
</style>

    """,

    unsafe_allow_html=True,

)



# =========================================================

# CLOUD SECRETS

# =========================================================

try:

    if st.secrets.get("GROQ_API_KEY"):

        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    if st.secrets.get("GROQ_MODEL"):

        os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]

except Exception:

    pass



# =========================================================

# HELPERS

# =========================================================

@st.cache_data

def read_csv_bytes(file_bytes):

    return pd.read_csv(BytesIO(file_bytes))





def validate_maintenance_dataframe(df):

    result = {"valid": True, "errors": [], "warnings": []}

    if df.empty:

        result["valid"] = False

        result["errors"].append("Maintenance CSV contains no data rows.")

        return result



    missing = [

        c for c in MAINTENANCE_REQUIRED_COLUMNS

        if c not in df.columns

    ]

    if missing:

        result["valid"] = False

        result["errors"].append(

            "Missing required maintenance columns: " + ", ".join(missing)

        )

        return result



    dates = pd.to_datetime(df["service_date"], errors="coerce")

    bad = int(dates.isna().sum())

    if bad:

        result["warnings"].append(

            f"{bad} maintenance record(s) have invalid service dates."

        )

    return result





def prepare_maintenance_dataframe(df):

    out = df.copy()

    out["service_date"] = pd.to_datetime(out["service_date"], errors="coerce")

    out["vehicle_id"] = out["vehicle_id"].astype(str).str.strip()

    return out





def count_fleet_anomalies(results):

    return sum(

        len(r.get("anomalies", []))

        for r in results

        if isinstance(r, dict)

    )





def hero(title, subtitle, kicker):

    st.markdown(

        f"""

        <div class="hero">

          <div class="hero-kicker">{kicker}</div>

          <h1>{title}</h1>

          <p>{subtitle}</p>

        </div>

        """,

        unsafe_allow_html=True,

    )





def status_pill(status):

    s = str(status)

    low = s.lower()

    if low == "good":

        cls, icon = "good", "●"

    elif low == "critical":

        cls, icon = "crit", "▲"

    else:

        cls, icon = "warn", "◆"

    st.markdown(

        f'<span class="pill {cls}">{icon} {s}</span>',

        unsafe_allow_html=True,

    )





def clean_label(v):

    return str(v).replace("_", " ").title()





def render_dict_clean(data):

    if not data:

        st.caption("No evidence available.")

        return

    if not isinstance(data, dict):

        st.write(data)

        return



    for key, value in data.items():

        label = clean_label(key)

        if isinstance(value, list):

            st.markdown(f"**{label}**")

            if not value:

                st.caption("None")

            for item in value:

                if isinstance(item, dict):

                    text = " | ".join(

                        f"**{clean_label(k)}:** {v}"

                        for k, v in item.items()

                    )

                    st.markdown(

                        f'<div class="evidence">{text}</div>',

                        unsafe_allow_html=True,

                    )

                else:

                    st.markdown(f"• {item}")

        elif isinstance(value, dict):

            st.markdown(f"**{label}**")

            for k, v in value.items():

                st.write(f"{clean_label(k)}: {v}")

        else:

            st.markdown(f"**{label}:** {value}")





def render_diagnosis_list(items):

    if not items:

        st.caption("No evidence available.")

        return

    if not isinstance(items, list):

        st.write(items)

        return



    for item in items:

        if not isinstance(item, dict):

            st.markdown(f"• {item}")

            continue



        title = (

            item.get("cause")

            or item.get("step")

            or item.get("issue")

            or item.get("component")

            or "Evidence item"

        )

        source = item.get("evidence_source")

        evidence = item.get("evidence")

        ref = item.get("source_reference")



        st.markdown(

            f"""

            <div class="evidence">

              <div class="card-title">{title}</div>

              <div class="muted tiny">{f"Evidence source: {source}" if source else ""}</div>

              <div>{evidence or ""}</div>

              <div class="muted tiny">{f"Reference: {ref}" if ref else ""}</div>

            </div>

            """,

            unsafe_allow_html=True,

        )



# =========================================================

# SIDEBAR — NAV FIRST, DATA SECOND

# =========================================================

with st.sidebar:

    st.markdown(

        """

        <div class="brand">

          <div class="brand-icon">🚛</div>

          <div>

            <div class="brand-title">FleetPredict AI</div>

            <div class="brand-sub">Fleet Diagnostic Co-Pilot</div>

          </div>

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)



    page = st.radio(

        "Navigation",

        [

            "🏠 Fleet Overview",

            "🚛 Vehicle Intelligence",

            "🛠 Maintenance History",

            "🤖 AI Diagnosis",

            "📋 Data & Schema",

        ],

        label_visibility="collapsed",

    )



    st.divider()



    st.markdown('<div class="side-label">Data Sources</div>', unsafe_allow_html=True)



    # Once sensor is uploaded, keep data section collapsed to preserve sidebar space.

    data_expanded = "sensor_upload" not in st.session_state or st.session_state.get("sensor_upload") is None



    with st.expander("Upload / change data", expanded=data_expanded):

        sensor_upload = st.file_uploader(

            "Sensor Telemetry CSV",

            type=["csv"],

            key="sensor_upload",

        )



        maintenance_upload = st.file_uploader(

            "Maintenance History CSV",

            type=["csv"],

            key="maintenance_upload",

            help="Optional but recommended for historical evidence.",

        )



    st.markdown('<div class="side-label">Knowledge Base</div>', unsafe_allow_html=True)

    st.success("✓ Isuzu F-Series Manual Ready")



    if st.session_state.get("maintenance_upload") is not None:

        st.success("✓ Maintenance History Loaded")

    else:

        st.info("Maintenance History Optional")



    st.divider()

    st.caption("Hackathon MVP • Streamlit • RAG • Groq")



# Pull uploads from session state

sensor_upload = st.session_state.get("sensor_upload")

maintenance_upload = st.session_state.get("maintenance_upload")



# =========================================================

# LANDING

# =========================================================

if sensor_upload is None:

    hero(

        "Smarter Fleet Decisions",

        "Upload sensor telemetry to monitor fleet health, detect anomalies, connect service history, and generate evidence-grounded AI diagnosis.",

        "FleetPredict AI",

    )



    left, right = st.columns([1.25, 1])



    with left:

        st.markdown("### What FleetPredict does")

        st.markdown(

            """

            <div class="card">

              <div class="card-title">One workflow from raw telemetry to action</div>

              <div class="muted">

                Validate uploaded data, calculate vehicle health, detect anomalies,

                connect maintenance history, retrieve repair-manual evidence, and

                generate a grounded AI diagnosis.

              </div>

            </div>

            """,

            unsafe_allow_html=True,

        )

        st.markdown(

            """

            **Operational workflow**

            - Fleet-wide health overview

            - Vehicle-level system monitoring and trends

            - Recurring maintenance patterns

            - Evidence-grounded AI diagnosis

            """

        )



    with right:

        st.markdown("### Required Sensor Schema")

        schema = [

            "vehicle_id", "timestamp", "engine_speed_rpm", "vehicle_speed_kmh",

            "engine_load_pct", "accelerator_pedal_pct", "coolant_temp_c",

            "oil_pressure_kpa", "fuel_rail_pressure_mpa", "battery_voltage_v",

            "maf_g_s",

        ]

        chips = "".join(

            f'<div class="schema-chip">{x}</div>' for x in schema

        )

        st.markdown(

            f'<div class="schema-grid">{chips}</div>',

            unsafe_allow_html=True,

        )



    st.stop()



# =========================================================

# SENSOR VALIDATION

# =========================================================

try:

    sensor_df = read_csv_bytes(sensor_upload.getvalue())

except Exception as e:

    st.error(f"Could not read sensor CSV: {e}")

    st.stop()



validation_result = validate_sensor_dataframe(sensor_df)



if not validation_result["valid"]:

    hero(

        "Sensor Data Validation",

        "The uploaded telemetry must pass validation before FleetPredict can continue.",

        "Validation Required",

    )

    for e in validation_result["errors"]:

        st.error(e)

    st.stop()



try:

    sensor_df = prepare_sensor_dataframe(sensor_df)

except Exception as e:

    st.error(f"Sensor preparation failed: {e}")

    st.stop()



# =========================================================

# MAINTENANCE

# =========================================================

maintenance_df = pd.DataFrame()



if maintenance_upload is not None:

    try:

        raw_maintenance_df = read_csv_bytes(maintenance_upload.getvalue())

        mv = validate_maintenance_dataframe(raw_maintenance_df)

        if not mv["valid"]:

            for e in mv["errors"]:

                st.error(e)

            st.stop()

        maintenance_df = prepare_maintenance_dataframe(raw_maintenance_df)

    except Exception as e:

        st.error(f"Could not process maintenance CSV: {e}")

        st.stop()



# =========================================================

# SHARED ANALYTICS

# =========================================================

try:

    fleet_health = calculate_fleet_health(sensor_df)

    fleet_anomalies = detect_fleet_anomalies(sensor_df)

    vehicles = get_available_vehicles(sensor_df)

except Exception as e:

    st.error(f"Fleet analytics failed: {e}")

    st.stop()



health_df = pd.DataFrame(fleet_health)



# =========================================================

# FLEET OVERVIEW

# =========================================================

if page == "🏠 Fleet Overview":

    hero(

        "Fleet Overview",

        "Operational health snapshot across the uploaded fleet — without scrolling through unrelated sections.",

        "Command Center",

    )



    counts = health_df["status"].value_counts()

    good = int(counts.get("Good", 0))

    warning = int(counts.get("Warning", 0))

    critical = int(counts.get("Critical", 0))

    anomalies = count_fleet_anomalies(fleet_anomalies)



    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Vehicles", len(health_df))

    m2.metric("Good", good)

    m3.metric("Warning", warning)

    m4.metric("Critical", critical)

    m5.metric("Anomalies", anomalies)



    left, right = st.columns([.95, 1.45])



    with left:

        chart_df = (

            health_df["status"]

            .value_counts()

            .rename_axis("Status")

            .reset_index(name="Vehicles")

        )



        fig = px.bar(

            chart_df,

            x="Status",

            y="Vehicles",

            color="Status",

            text="Vehicles",

            color_discrete_map={

                "Good": C["teal"],

                "Warning": C["amber"],

                "Critical": C["red"],

            },

            title="Health Distribution",

        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor=C["panel"],

            plot_bgcolor=C["panel"],

            font_color=C["text"],

            showlegend=False,

            height=410,

            margin=dict(l=18, r=18, t=55, b=18),

        )

        st.plotly_chart(fig, width="stretch")



    with right:

        st.markdown("### Vehicle Health Summary")

        rows = []

        for v in fleet_health:

            s = v.get("systems", {})

            rows.append(

                {

                    "Vehicle": v.get("vehicle_id"),

                    "Health": v.get("overall_health"),

                    "Status": v.get("status"),

                    "Engine": s.get("engine"),

                    "Cooling": s.get("cooling"),

                    "Oil": s.get("oil"),

                    "Fuel": s.get("fuel"),

                    "Electrical": s.get("electrical"),

                    "Anomalies": v.get("anomaly_count", 0),

                }

            )

        st.dataframe(

            pd.DataFrame(rows),

            width="stretch",

            hide_index=True,

            height=410,

        )



# =========================================================

# VEHICLE DATA FOR 3 PAGES

# =========================================================

if page in [

    "🚛 Vehicle Intelligence",

    "🛠 Maintenance History",

    "🤖 AI Diagnosis",

]:

    selected_vehicle = st.selectbox(

        "Selected Vehicle",

        vehicles,

        key=f"selected_{page}",

    )



    try:

        vehicle_records = get_vehicle_records(sensor_df, selected_vehicle)

        latest_reading = get_latest_vehicle_reading(sensor_df, selected_vehicle)

        vehicle_health = calculate_vehicle_health(sensor_df, selected_vehicle)

        vehicle_anomalies = detect_vehicle_anomalies(sensor_df, selected_vehicle)

    except Exception as e:

        st.error(f"Vehicle analysis failed: {e}")

        st.stop()



# =========================================================

# VEHICLE INTELLIGENCE

# =========================================================

if page == "🚛 Vehicle Intelligence":

    hero(

        "Vehicle Intelligence",

        f"Latest health, system status, sensor readings, anomalies, and trends for {selected_vehicle}.",

        "Vehicle Detail",

    )



    a, b, c = st.columns(3)

    a.metric("Overall Health", vehicle_health.get("overall_health", "N/A"))

    with b:

        st.caption("Health Status")

        status_pill(vehicle_health.get("status", "Unknown"))

    c.metric("Active Anomalies", vehicle_health.get("anomaly_count", 0))



    st.markdown("### System Health")

    systems = vehicle_health.get("systems", {})

    cols = st.columns(max(len(systems), 1))

    for col, (name, score) in zip(cols, systems.items()):

        col.metric(name.title(), score)



    st.markdown("### Latest Sensor Readings")

    readings = {

        "Engine RPM": latest_reading.get("engine_speed_rpm"),

        "Speed km/h": latest_reading.get("vehicle_speed_kmh"),

        "Coolant °C": latest_reading.get("coolant_temp_c"),

        "Oil kPa": latest_reading.get("oil_pressure_kpa"),

        "Fuel Rail MPa": latest_reading.get("fuel_rail_pressure_mpa"),

        "Battery V": latest_reading.get("battery_voltage_v"),

    }

    cols = st.columns(3)

    for i, (label, value) in enumerate(readings.items()):

        cols[i % 3].metric(label, value)



    left, right = st.columns([1.25, 1])



    with left:

        st.markdown("### Sensor Trend")

        trend_options = {

            "Coolant Temperature": "coolant_temp_c",

            "Oil Pressure": "oil_pressure_kpa",

            "Battery Voltage": "battery_voltage_v",

            "Engine RPM": "engine_speed_rpm",

            "Fuel Rail Pressure": "fuel_rail_pressure_mpa",

        }

        trend_name = st.selectbox("Sensor", trend_options.keys())

        trend_col = trend_options[trend_name]



        fig = px.line(

            vehicle_records,

            x="timestamp",

            y=trend_col,

            title=trend_name,

        )

        fig.update_traces(line=dict(color=C["cyan"], width=2.7))

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor=C["panel"],

            plot_bgcolor=C["panel"],

            font_color=C["text"],

            height=365,

            margin=dict(l=15, r=15, t=50, b=15),

        )

        st.plotly_chart(fig, width="stretch")



    with right:

        st.markdown("### Detected Anomalies")

        anomalies = vehicle_anomalies.get("anomalies", [])



        if not anomalies:

            st.success("✓ No active anomalies detected.")

        else:

            for anomaly in anomalies:

                issue = anomaly.get("issue", "Detected anomaly")

                severity = anomaly.get("severity", "Warning")

                metric = anomaly.get("metric", "")

                value = anomaly.get("value", "")

                source = anomaly.get("rule_source", "")



                msg = f"**{issue}**\n\n{metric}: {value}\n\nSource: {source}"

                st.error(msg) if severity == "Critical" else st.warning(msg)



# =========================================================

# MAINTENANCE HISTORY

# =========================================================

if page == "🛠 Maintenance History":

    hero(

        "Maintenance History",

        f"Service evidence, recurring systems, repeated issues, and recent repairs for {selected_vehicle}.",

        "Service Intelligence",

    )



    if maintenance_df.empty:

        st.info(

            "No maintenance history CSV uploaded. Open **Upload / change data** "

            "in the sidebar and add the maintenance file."

        )

    else:

        try:

            summary = build_maintenance_summary(

                maintenance_df,

                selected_vehicle,

            )

            history = get_vehicle_history(

                maintenance_df,

                selected_vehicle,

            )

        except Exception as e:

            st.error(f"Maintenance analysis failed: {e}")

            st.stop()



        a, b = st.columns(2)

        a.metric("Maintenance Records", summary.get("total_records", 0))

        b.metric("Latest Service", summary.get("latest_service_date") or "No History")



        if summary.get("total_records", 0) == 0:

            st.info(f"No maintenance records found for {selected_vehicle}.")

        else:

            left, right = st.columns(2)



            with left:

                st.markdown("### Repeated Systems")

                repeated = summary.get("repeated_systems", [])

                if not repeated:

                    st.caption("No repeated systems.")

                for item in repeated:

                    st.markdown(

                        f"""

                        <div class="card">

                          <div class="card-title">{item.get("system")}</div>

                          <div class="muted tiny">{item.get("count")} maintenance records</div>

                        </div>

                        """,

                        unsafe_allow_html=True,

                    )



            with right:

                st.markdown("### Repeated Issues")

                repeated = summary.get("repeated_issues", [])

                if not repeated:

                    st.caption("No repeated issues.")

                for item in repeated:

                    st.markdown(

                        f"""

                        <div class="card">

                          <div class="card-title">{item.get("issue")}</div>

                          <div class="muted tiny">{item.get("count")} records</div>

                        </div>

                        """,

                        unsafe_allow_html=True,

                    )



            st.markdown("### Recent Maintenance")

            recent = pd.DataFrame(summary.get("recent_repairs", []))



            if not recent.empty:

                preferred = [

                    "service_date", "system", "detected_issue",

                    "probable_cause", "repair_action", "component", "status",

                ]

                existing = [x for x in preferred if x in recent.columns]

                st.dataframe(

                    recent[existing],

                    width="stretch",

                    hide_index=True,

                )



            with st.expander("View complete service history"):

                st.dataframe(

                    history,

                    width="stretch",

                    hide_index=True,

                )



# =========================================================

# AI DIAGNOSIS

# =========================================================

if page == "🤖 AI Diagnosis":

    hero(

        "AI Diagnostic Co-Pilot",

        f"Generate an evidence-grounded diagnosis for {selected_vehicle} using telemetry, maintenance history, and retrieved Isuzu manual context.",

        "Grounded AI",

    )



    a, b, c = st.columns(3)

    a.metric("Health", vehicle_health.get("overall_health"))

    with b:

        st.caption("Vehicle Status")

        status_pill(vehicle_health.get("status", "Unknown"))

    c.metric("Current Anomalies", vehicle_health.get("anomaly_count", 0))



    if maintenance_df.empty:

        st.info(

            "Maintenance history was not supplied. Diagnosis will rely on "

            "sensor and repair-manual evidence."

        )



    if not os.getenv("GROQ_API_KEY"):

        st.warning("Groq API key is unavailable.")



    if st.button(

        "✨ Generate AI Diagnosis",

        type="primary",

        use_container_width=True,

    ):

        with st.spinner(

            "Retrieving manual evidence and generating grounded diagnosis..."

        ):

            result = run_vehicle_diagnosis(

                sensor_df,

                maintenance_df,

                selected_vehicle,

            )



        if not result.get("success"):

            st.error(result.get("error", "Diagnosis failed."))

        else:

            st.session_state[

                f"diagnosis_{selected_vehicle}"

            ] = result.get("diagnosis", {})



    diagnosis = st.session_state.get(f"diagnosis_{selected_vehicle}")



    if diagnosis:

        st.success("✓ Diagnosis completed")



        x, y = st.columns(2)

        with x:

            st.caption("Diagnosis Severity")

            status_pill(diagnosis.get("severity", "Unknown"))

        y.metric("Vehicle", selected_vehicle)



        st.markdown("### Problem")

        st.markdown(

            f"""

            <div class="card">

              <div class="card-title">{diagnosis.get("problem","No summary returned.")}</div>

            </div>

            """,

            unsafe_allow_html=True,

        )



        st.markdown("### Health Summary")

        st.write(diagnosis.get("health_summary", "Not available."))



        tab1, tab2, tab3, tab4 = st.tabs(

            [

                "📡 Sensor Evidence",

                "🛠 Maintenance Evidence",

                "🔎 Possible Causes",

                "📖 Inspection & Manual",

            ]

        )



        with tab1:

            render_dict_clean(diagnosis.get("sensor_evidence", {}))



        with tab2:

            if maintenance_df.empty:

                st.info("No maintenance history was supplied.")

            else:

                render_dict_clean(diagnosis.get("historical_pattern", {}))



        with tab3:

            render_diagnosis_list(diagnosis.get("possible_causes", []))



        with tab4:

            st.markdown("### Evidence-Based Inspection Steps")

            render_diagnosis_list(diagnosis.get("inspection_steps", []))



            st.markdown("### Relevant Components")

            render_diagnosis_list(diagnosis.get("relevant_components", []))



            st.markdown("### Repair Manual Sources")

            sources = diagnosis.get("manual_sources", [])

            if sources:

                for source in sources:

                    if isinstance(source, dict):

                        st.markdown(f"• Page {source.get('page','Unknown')}")

                    else:

                        st.markdown(f"• Page {source}")

            else:

                st.caption("No manual sources returned.")



        st.markdown("### Safety Action")

        st.warning(diagnosis.get("safety_action", "No safety action returned."))



        st.markdown("### Confidence Note")

        st.info(diagnosis.get("confidence_note", "Not available."))

    else:

        st.markdown(

            """

            <div class="card">

              <div class="card-title">Ready for diagnosis</div>

              <div class="muted">

                Click Generate AI Diagnosis to combine health, anomalies,

                maintenance evidence, and retrieved repair-manual context.

              </div>

            </div>

            """,

            unsafe_allow_html=True,

        )



# =========================================================

# DATA & SCHEMA

# =========================================================

if page == "📋 Data & Schema":

    hero(

        "Data & Schema",

        "Review uploaded datasets, validation status, and the FleetPredict input contract.",

        "Data Quality",

    )



    a, b, c = st.columns(3)

    a.metric("Sensor Records", len(sensor_df))

    b.metric("Vehicles", sensor_df["vehicle_id"].nunique())

    c.metric("Maintenance Records", len(maintenance_df))



    st.success("✓ Sensor schema validated successfully.")



    if validation_result["warnings"]:

        with st.expander("Sensor validation warnings"):

            for w in validation_result["warnings"]:

                st.warning(w)



    with st.expander("Preview uploaded sensor data"):

        st.dataframe(sensor_df.head(30), width="stretch")



    st.markdown("### Required Sensor Schema")

    schema = [

        "vehicle_id", "timestamp", "engine_speed_rpm", "vehicle_speed_kmh",

        "engine_load_pct", "accelerator_pedal_pct", "coolant_temp_c",

        "oil_pressure_kpa", "fuel_rail_pressure_mpa", "battery_voltage_v",

        "maf_g_s",

    ]

    chips = "".join(f'<div class="schema-chip">{x}</div>' for x in schema)

    st.markdown(f'<div class="schema-grid">{chips}</div>', unsafe_allow_html=True)



    st.markdown("### Maintenance Data")

    if maintenance_df.empty:

        st.info("No maintenance CSV uploaded.")

    else:

        st.success("✓ Maintenance CSV validated.")

        with st.expander("Preview maintenance data"):

            st.dataframe(maintenance_df.head(30), width="stretch")



# =========================================================

# FOOTER

# =========================================================

st.markdown(

    """

    <div class="footer">

      FleetPredict AI • Hackathon MVP • Sensor Analytics + Maintenance + RAG + Groq

    </div>

    """,

    unsafe_allow_html=True,

)
