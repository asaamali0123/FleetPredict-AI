import json
import os

from dotenv import load_dotenv
from groq import Groq

from src.maintenance_service import build_maintenance_summary
from src.rag_retriever import retrieve_context


load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def create_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def get_groq_model():
    return os.getenv(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL,
    )

def test_groq_connection():
    client = create_groq_client()

    if client is None:
        return {
            "success": False,
            "error": "GROQ_API_KEY is missing.",
        }

    try:
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: FleetPredict AI connection successful",
                }
            ],
            temperature=0,
        )

        return {
            "success": True,
            "response": response.choices[0].message.content,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }

def build_diagnosis_prompt(
    health_result,
    anomaly_result,
    maintenance_summary,
    rag_result,
):
    prompt = f"""
You are FleetPredict AI, a fleet diagnostic assistant.

Your task is to analyze the provided vehicle evidence and produce a
grounded maintenance diagnosis.

IMPORTANT RULES:
- Use only the evidence provided below.
- Clearly distinguish sensor evidence, maintenance-history evidence,
  manual evidence, and AI inference.
- Maintenance history is supporting evidence only. It is NOT proof
  that the same fault is currently present.
- Do not invent OEM thresholds, specifications, part numbers,
  manual pages, or repair procedures.
- Do not recommend unsafe operation when the vehicle has a Critical fault.
- If the retrieved manual context says:
  "Not found in retrieved manual context."
  then explicitly state:
  "Relevant procedure was not found in the retrieved manual context."

VEHICLE:
{health_result.get("vehicle_id")}

HEALTH RESULT:
{health_result}

ANOMALY RESULT:
{anomaly_result}

MAINTENANCE HISTORY:
{maintenance_summary}

RETRIEVED MANUAL CONTEXT:
{rag_result.get("context", "Not found in retrieved manual context.")}

MANUAL SOURCES:
{rag_result.get("sources", [])}

Return a concise diagnosis containing:
1. Problem
2. Severity
3. Health summary
4. Sensor evidence
5. Historical pattern
6. Possible causes
7. Safety action
8. Inspection steps
9. Relevant components
10. Manual sources
11. Confidence note

Do not claim that a possible cause is confirmed unless the evidence
directly supports it.
"""

    return prompt

def request_ai_diagnosis(prompt):
    client = create_groq_client()

    if client is None:
        return {
            "success": False,
            "error": "GROQ_API_KEY is missing.",
        }

    try:
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful vehicle diagnostic assistant. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        return {
            "success": True,
            "response": response.choices[0].message.content,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }

def parse_ai_response(response_text):
    if not response_text:
        return {
            "success": False,
            "error": "AI returned an empty response.",
        }

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": f"AI returned invalid JSON: {exc}",
        }

    if not isinstance(parsed, dict):
        return {
            "success": False,
            "error": "AI response must be a JSON object.",
        }

    return {
        "success": True,
        "data": parsed,
    }

def run_vehicle_diagnosis(
    dataframe,
    maintenance_df,
    vehicle_id,
):
    from src.anomaly_detector import detect_vehicle_anomalies
    from src.health_score import calculate_vehicle_health

    try:
        # Member 1: anomaly detection
        anomaly_result = detect_vehicle_anomalies(
            dataframe,
            vehicle_id,
        )

        # Member 1: health score
        health_result = calculate_vehicle_health(
            dataframe,
            vehicle_id,
        )

        # Member 3: maintenance history
        maintenance_summary = build_maintenance_summary(
            maintenance_df,
            vehicle_id,
        )

        # Member 2: RAG/manual context
        anomalies = anomaly_result.get("anomalies", [])

        if anomalies:
            rag_query = " ".join(
                anomaly.get("issue", "")
                for anomaly in anomalies
                if anomaly.get("issue")
            )
        else:
            rag_query = "vehicle maintenance inspection"

        rag_result = retrieve_context(rag_query)

        # Build grounded AI prompt
        prompt = build_diagnosis_prompt(
            health_result,
            anomaly_result,
            maintenance_summary,
            rag_result,
        )

        # Ask Groq
        ai_result = request_ai_diagnosis(prompt)

        if not ai_result["success"]:
            return ai_result

        # Convert JSON string to Python dictionary
        parsed_result = parse_ai_response(
            ai_result["response"]
        )

        if not parsed_result["success"]:
            return parsed_result

        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "diagnosis": parsed_result["data"],
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }