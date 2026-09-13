# 🚛 FleetPredict AI

**AI-Powered Diagnostic Co-Pilot for Commercial Vehicle Fleets**

[🌐 Live Demo](https://fleetpredict-ai.streamlit.app/) · Built for the
**PAKANGELS Mid-term Hackathon**

> **Rules detect → RAG retrieves → AI explains.**

FleetPredict AI is a Streamlit-based diagnostic co-pilot that combines
vehicle telemetry, deterministic health scoring, anomaly detection,
maintenance history, repair-manual retrieval, and generative AI to turn
raw fleet data into actionable maintenance guidance.

------------------------------------------------------------------------

## 🎯 Problem

Commercial fleets already generate large amounts of vehicle data, but
converting that data into maintenance decisions can still take
significant time.

Fleet teams may need to interpret sensor readings, review scattered
service records, and search through hundreds of pages of technical
manuals before deciding what to inspect next.

FleetPredict AI brings those sources together in one workflow.

------------------------------------------------------------------------

## 💡 Solution

FleetPredict AI analyzes uploaded vehicle telemetry before involving the
LLM.

The system:

-   validates incoming sensor data;
-   calculates overall and subsystem health;
-   detects anomalies using deterministic rules;
-   adds vehicle-specific maintenance history;
-   retrieves relevant evidence from the repair manual using RAG;
-   sends the grounded context to a Groq-powered LLM;
-   returns a structured diagnostic explanation.

This keeps **detection and calculation in Python**, while AI is used to
**explain and organize evidence**.

------------------------------------------------------------------------

## 🌐 Live Application

### **[Launch FleetPredict AI](https://fleetpredict-ai.streamlit.app/)**

The deployed MVP allows users to upload compatible telemetry data,
optionally add maintenance history, explore fleet and vehicle health,
and generate an evidence-grounded AI diagnosis.

------------------------------------------------------------------------

## ⚙️ How It Works

``` text
Sensor Telemetry CSV          Maintenance History CSV (Optional)
          │                              │
          ▼                              │
   Schema Validation                     │
          │                              │
          ▼                              │
   Sensor Analytics                      │
          │                              │
     ┌────┴─────┐                        │
     ▼          ▼                        │
Health Score  Anomaly Detection          │
     │          │                        │
     └────┬─────┘                        │
          ▼                              ▼
          Vehicle Diagnostic Context
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 Maintenance Evidence    Repair Manual RAG
                              │
                    PDF → Chunks → Embeddings
                              │
                             FAISS
                              │
                    Relevant Manual Evidence
          └─────────┬─────────┘
                    ▼
             Grounded Context
                    │
                    ▼
             Groq-powered LLM
                    │
                    ▼
          Structured AI Diagnosis
```

**Python detects. RAG retrieves. AI explains.**

------------------------------------------------------------------------

## ✨ Key Features

### 📥 Sensor Upload & Schema Validation

FleetPredict accepts telemetry CSV files that follow a predefined
schema. Before analytics run, the application checks required columns,
timestamps, numeric sensor fields, duplicate columns, empty datasets,
and vehicle identifiers.

### 📊 Fleet Overview

The dashboard provides a fleet-level snapshot with vehicle counts,
Good/Warning/Critical statuses, health scores, subsystem health, and
anomaly counts.

### 🚛 Vehicle Intelligence

Users can inspect an individual vehicle's latest sensor readings,
overall health, engine/cooling/oil/fuel/electrical health, sensor
trends, and active anomalies.

### ❤️ Vehicle Health Scoring

Rule-based logic converts current telemetry into an overall health score
and subsystem scores, helping users quickly identify areas that need
attention.

### ⚠️ Deterministic Anomaly Detection

Anomalies are detected using Python rules rather than asking the LLM to
decide whether raw sensor values are abnormal.

### 🛠️ Maintenance History

An optional maintenance CSV adds previous service records, detected
issues, repair actions, components, and repeated-problem context to the
diagnostic workflow.

### 📚 Repair Manual RAG

The Isuzu F-Series service manual is processed into searchable chunks.
Sentence embeddings and FAISS are used to retrieve relevant technical
evidence for the current vehicle problem.

### 🤖 AI Diagnostic Co-Pilot

FleetPredict combines sensor evidence, health scores, anomalies,
maintenance context, and retrieved manual information before generating
a structured diagnosis with the Groq-powered LLM.

------------------------------------------------------------------------

## 📋 Required Telemetry Schema

A sensor CSV must contain these fields:

  Field                      Description
  -------------------------- ----------------------------
  `vehicle_id`               Unique vehicle identifier
  `timestamp`                Sensor reading timestamp
  `engine_speed_rpm`         Engine speed
  `vehicle_speed_kmh`        Vehicle speed
  `engine_load_pct`          Engine load percentage
  `accelerator_pedal_pct`    Accelerator pedal position
  `coolant_temp_c`           Coolant temperature
  `oil_pressure_kpa`         Oil pressure
  `fuel_rail_pressure_mpa`   Fuel rail pressure
  `battery_voltage_v`        Battery voltage
  `maf_g_s`                  Mass air flow

Optional supported fields include `charging_status` and `dtc_flag`.

------------------------------------------------------------------------

## 🧰 Tech Stack

  Layer             Technology
  ----------------- ----------------------------------
  Programming       Python
  Data Processing   Pandas, NumPy
  Web App           Streamlit
  Visualization     Plotly
  PDF Processing    PyPDF
  Embeddings        Sentence Transformers
  Vector Search     FAISS
  Generative AI     Groq API
  RAG               Custom Python retrieval pipeline
  Deployment        Streamlit Community Cloud
  Version Control   Git & GitHub

------------------------------------------------------------------------

## 📁 Project Structure

``` text
FleetPredict-AI/
├── data/
│   ├── maintenance/
│   ├── manuals/
│   └── sample/
├── docs/
│   └── team_handoff.md
├── src/
│   ├── ai_diagnosis.py
│   ├── anomaly_detector.py
│   ├── health_score.py
│   ├── maintenance_service.py
│   ├── manual_processor.py
│   ├── rag_retriever.py
│   ├── schema_validator.py
│   ├── sensor_analytics.py
│   └── __init__.py
├── tests/
├── vector_store/
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 🚀 Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/asaamali0123/FleetPredict-AI.git
cd FleetPredict-AI
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

On Windows:

``` powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure the Groq API

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Never commit your real API key.

### 5. Run Streamlit

``` bash
python -m streamlit run streamlit_app.py
```

------------------------------------------------------------------------

## 🧪 Hackathon Demo Data

The MVP includes prepared Isuzu 6HK1 demonstration data:

-   **30 vehicles**
-   **3,600 telemetry records**
-   **96 maintenance records**
-   normal and injected fault scenarios
-   an Isuzu F-Series service manual used as the RAG knowledge base

These files are for development and hackathon demonstration purposes,
not live production telemetry.

------------------------------------------------------------------------

## 🧠 Design Principle

FleetPredict deliberately separates analytical logic from generative
reasoning:

``` text
Python / Pandas
      ↓
Calculate & Detect

RAG / FAISS
      ↓
Retrieve Evidence

Groq-powered LLM
      ↓
Explain & Structure
```

The LLM is a **diagnostic co-pilot**, not the source of sensor
measurements, health calculations, or anomaly rules.

------------------------------------------------------------------------

## 🔮 Future Scope

The current version is a focused hackathon MVP. Future development could
include:

-   real-time IoT telemetry integration;
-   predictive machine-learning failure models;
-   support for multiple vehicle manufacturers and manuals;
-   real-time alerts and notifications;
-   automated maintenance scheduling;
-   persistent fleet databases and historical analytics;
-   production authentication and role-based access.

> **Today: Detect & Explain → Tomorrow: Predict Before Failure**

------------------------------------------------------------------------

## 👥 Team

### FleetPredict AI --- PAKANGELS Mid-term Hackathon

-   **Muhammad Assam**
-   **Wajeeha Fatima**
-   **Nimra Shahnawaz**
-   **Muhammad Mehrooz**

------------------------------------------------------------------------

## ⚠️ Disclaimer

FleetPredict AI is a hackathon prototype built for demonstration and
educational purposes. Its health scores, anomaly rules, and AI-generated
guidance should not replace manufacturer procedures, qualified
mechanical inspection, or professional safety decisions.

------------------------------------------------------------------------

## ⭐ Why FleetPredict AI?

FleetPredict AI demonstrates a practical combination of **data
analytics + deterministic diagnostics + RAG + generative AI**.

Instead of building a generic chatbot, the system first understands the
vehicle:

**Telemetry → Validation → Analytics → Health → Anomalies → Maintenance
→ RAG → AI Diagnosis**

Then it asks AI to explain the evidence.
