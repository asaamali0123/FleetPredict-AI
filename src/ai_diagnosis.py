import json
import os

from dotenv import load_dotenv
from groq import Groq

from src.maintenance_service import build_maintenance_summary
from src.rag_retriever import retrieve_context


load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


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
                    "content": (
                        "Reply with exactly: "
                        "FleetPredict AI connection successful"
                    ),
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
You are FleetPredict AI, a careful fleet diagnostic assistant.

Your task is to analyze ONLY the evidence provided below and produce a
grounded maintenance diagnosis.

IMPORTANT GROUNDING RULES:

- Use only the evidence provided below.

- Clearly distinguish:
  1. sensor evidence
  2. maintenance-history evidence
  3. retrieved manual evidence
  4. AI inference

- Maintenance history is supporting evidence only.
  It is NOT proof that the same fault is currently present.

- Never strengthen uncertain maintenance wording.
  If history says "suspected", "possible", "inspection",
  or similar uncertainty, preserve that uncertainty.

- Never convert a suspected maintenance cause into a more specific
  mechanical failure mode.

  Example:
  If history says:
  "Thermostat or coolant circulation problem suspected"

  Do NOT change it to:
  "Thermostat stuck closed"

- Do not invent:
  OEM thresholds,
  specifications,
  part numbers,
  manual pages,
  repair procedures,
  inspection procedures,
  component failures,
  replacement instructions,
  adjustment instructions,
  bleeding procedures,
  or topping-up procedures.

- Possible causes may only be stated as evidence-backed causes when they
  are directly supported by sensor evidence, maintenance history, or
  retrieved manual context.

- If a possible cause is not directly supported by supplied evidence,
  label it explicitly as:
  "General AI inference - not confirmed by provided evidence."

- When converting a maintenance-history repair_action into an inspection
  step, preserve its meaning closely.

- Do NOT add any action, condition, replacement, adjustment, bleeding,
  topping-up, repair, or failure mode that is not explicitly present in
  the maintenance record.

  Example:
  If maintenance history says:
  "Inspect thermostat"

  Do NOT change it to:
  "Inspect thermostat and replace if stuck"

- Inspection steps must come ONLY from:
  1. retrieved manual context, or
  2. an explicit repair_action from maintenance history.

- Do not create general mechanical inspection procedures from your own
  knowledge.

- Do not describe a raw sensor value as:
  "high",
  "low",
  "abnormal",
  "outside normal range",
  "above specification",
  or similar wording

  UNLESS that classification is already present in ANOMALY RESULT or is
  directly supported by RETRIEVED MANUAL CONTEXT.

- A raw value may always be reported factually.

  Example:
  GOOD:
  "Coolant temperature reading is 100.9 C."

  NOT ALLOWED unless supported:
  "Coolant temperature 100.9 C exceeds the normal range."

- Health classifications and anomaly labels produced by the existing
  FleetPredict rules may be reported exactly as supplied.

- Relevant components must come only from:
  sensor/anomaly evidence,
  maintenance history,
  or retrieved manual context.

- Do not add related vehicle components using general mechanical
  knowledge.

- If no supplied evidence supports a specific inspection procedure, say:
  "No specific procedure found in retrieved manual context."

- A manual page number may only be cited if that exact page appears in
  MANUAL SOURCES below.

- Never create, estimate, or infer a manual page number.

- Do not claim that a possible cause is confirmed unless the evidence
  directly confirms it.

- Do not recommend unsafe vehicle operation for serious faults.

- Safety advice should remain conservative and should not introduce
  unsupported technical thresholds or procedures.

- If retrieved manual context is unavailable, explicitly state:
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

Return valid JSON only.

Return a concise diagnosis using these fields:

{{
  "problem": "...",
  "severity": "...",
  "health_summary": "...",
  "sensor_evidence": {{}},
  "historical_pattern": {{}},
  "possible_causes": [
    {{
      "cause": "...",
      "evidence_source": "sensor | maintenance_history | manual | ai_inference",
      "evidence": "..."
    }}
  ],
  "safety_action": "...",
  "inspection_steps": [
    {{
      "step": "...",
      "evidence_source": "maintenance_history | manual",
      "source_reference": "..."
    }}
  ],
  "relevant_components": [],
  "manual_sources": [],
  "confidence_note": "..."
}}

For possible causes and inspection steps, always identify their evidence
source.

For maintenance-history inspection steps, stay very close to the original
repair_action wording.

For manual inspection steps, use only procedures explicitly present in
the retrieved text.

Copy manual sources only from MANUAL SOURCES supplied above.

Do not create new page numbers.

Do not make a diagnosis sound more certain or more specific than the
provided evidence supports.
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
                        "You are a careful evidence-grounded vehicle "
                        "diagnostic assistant. Return valid JSON only. "
                        "Never invent technical procedures, specifications, "
                        "manual references, or confirmed causes. "
                        "Never expand supplied maintenance actions into "
                        "additional procedures. "
                        "Never strengthen suspected causes into specific "
                        "failure modes. "
                        "Never classify raw sensor values using unstated "
                        "thresholds. "
                        "Preserve uncertainty exactly when the evidence "
                        "is uncertain."
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

            if not rag_query.strip():
                rag_query = "vehicle maintenance inspection"
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