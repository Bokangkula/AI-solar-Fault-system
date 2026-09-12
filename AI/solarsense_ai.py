import os
import joblib
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()  # picks up GEMINI_API_KEY saved by run.py, if present

# ==============================
# LOAD ML MODEL
# ==============================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "solar_fault_model_engineered.pkl"
)
model = joblib.load(MODEL_PATH)

fault_names = {
    0: "Normal operation",
    1: "Short circuit",
    2: "Degradation",
    3: "Open circuit",
    4: "Shadowing"
}

# ==============================
# GET USER MEASUREMENTS
# ==============================

print("================================")
print("        SOLARSENSE AI")
print("================================")
print("Enter PV system measurements\n")

vdc1 = float(input("DC Voltage - String 1: "))
vdc2 = float(input("DC Voltage - String 2: "))
idc1 = float(input("DC Current - String 1: "))
idc2 = float(input("DC Current - String 2: "))
irr = float(input("Irradiance: "))
pvt = float(input("Temperature: "))

# ==============================
# ENGINEERED FEATURES
# ==============================

power1 = vdc1 * idc1
power2 = vdc2 * idc2

voltage_diff = vdc1 - vdc2
current_diff = idc1 - idc2
power_diff = power1 - power2

voltage_ratio = vdc1 / (vdc2 + 1e-6)
current_ratio = idc1 / (idc2 + 1e-6)
power_ratio = power1 / (power2 + 1e-6)

# ==============================
# CREATE MODEL INPUT
# ==============================

data = pd.DataFrame([{
    "vdc1": vdc1,
    "vdc2": vdc2,
    "idc1": idc1,
    "idc2": idc2,
    "irr": irr,
    "pvt": pvt,
    "power1": power1,
    "power2": power2,
    "voltage_diff": voltage_diff,
    "current_diff": current_diff,
    "power_diff": power_diff,
    "voltage_ratio": voltage_ratio,
    "current_ratio": current_ratio,
    "power_ratio": power_ratio
}])

# ==============================
# ML PREDICTION
# ==============================

prediction = model.predict(data)[0]

probabilities = model.predict_proba(data)[0]

confidence = max(probabilities) * 100

fault = fault_names[prediction]

print("\n================================")
print("       SOLARSENSE RESULT")
print("================================")

print("Detected condition:", fault)
print(f"Model confidence: {confidence:.2f}%")

# ==============================
# GEMINI
# ==============================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

prompt = f"""
You are the explanation layer of SolarSense, an AI-powered
solar PV fault-triage system.

A machine-learning model analyzed measurements from a PV system.

Predicted condition:
{fault}

Model confidence:
{confidence:.2f}%

Measurements:
DC voltage string 1: {vdc1}
DC voltage string 2: {vdc2}
DC current string 1: {idc1}
DC current string 2: {idc2}
Irradiance: {irr}
Temperature: {pvt}

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
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print("\n================================")
print("       AI EXPLANATION")
print("================================")

print(response.text)