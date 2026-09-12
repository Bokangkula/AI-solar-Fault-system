"""
Gemini integration - matches the SDK and model already used in
AI/solarsense_ai.py (the `google-genai` package, not `google-generativeai`).

Set your API key as an environment variable before running:
    export GEMINI_API_KEY=your_key_here        (Mac/Linux)
    setx GEMINI_API_KEY "your_key_here"         (Windows)

Get a key at https://aistudio.google.com/app/apikey

If no key is set, or the call fails for any reason (no wifi at the venue,
rate limit, etc.), this falls back to a canned explanation so the demo
never breaks on stage.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # picks up GEMINI_API_KEY from a local .env file, if present

DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

FALLBACK_TEMPLATE = (
    "Detected condition: {fault_label} ({confidence_pct}% confidence).\n\n"
    "This is a preliminary, ML-based triage result, not a certified diagnosis. "
    "A technician should inspect the relevant string, wiring, or panel surface "
    "before taking action.\n\n"
    "(Live AI explanation unavailable right now - showing a general fallback message.)"
)


def get_explanation(fault_label: str, confidence: float, readings: dict, api_key: str = None) -> str:
    confidence_pct = round(confidence * 100, 2)
    key = api_key or DEFAULT_API_KEY

    if not key:
        return FALLBACK_TEMPLATE.format(fault_label=fault_label, confidence_pct=confidence_pct)

    try:
        from google import genai

        client = genai.Client(api_key=key)

        prompt = f"""
You are the explanation layer of SolarSense, an AI-powered
solar PV fault-triage system.

A machine-learning model analyzed measurements from a PV system.

Predicted condition:
{fault_label}

Model confidence:
{confidence_pct}%

Measurements:
DC voltage string 1: {readings['vdc1']}
DC voltage string 2: {readings['vdc2']}
DC current string 1: {readings['idc1']}
DC current string 2: {readings['idc2']}
Irradiance: {readings['irr']}
Temperature: {readings['pvt']}

Explain the ML result for a solar technician.

Provide:
1. What the detected condition means
2. Possible causes
3. What the technician should inspect first

Keep the explanation concise and practical.

Do not claim certainty. This is a fault-triage system,
not a replacement for professional electrical inspection.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    except Exception as e:  # noqa: BLE001
        print(f"Gemini call failed, using fallback explanation: {e}")
        return FALLBACK_TEMPLATE.format(fault_label=fault_label, confidence_pct=confidence_pct)
