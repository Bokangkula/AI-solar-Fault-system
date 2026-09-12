"""
Prediction pipeline for SolarSense.

Wraps the trained model in models/solar_fault_model_engineered.pkl and
reproduces the exact feature engineering used at training time (see
AI/solarsense_ai.py for the original CLI version this was built from).

Model input features (must stay in this order - it's what the model
was trained on):
    vdc1, vdc2, idc1, idc2, irr, pvt,
    power1, power2,
    voltage_diff, current_diff, power_diff,
    voltage_ratio, current_ratio, power_ratio
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "solar_fault_model_engineered.pkl"
)

FAULT_NAMES = {
    0: "normal",
    1: "short_circuit",
    2: "degradation",
    3: "open_circuit",
    4: "shadowing",
}

_model = None
if os.path.exists(MODEL_PATH):
    _model = joblib.load(MODEL_PATH)


def build_features(vdc1: float, vdc2: float, idc1: float, idc2: float, irr: float, pvt: float) -> pd.DataFrame:
    """Reproduces the exact engineered features the model was trained on."""
    power1 = vdc1 * idc1
    power2 = vdc2 * idc2

    voltage_diff = vdc1 - vdc2
    current_diff = idc1 - idc2
    power_diff = power1 - power2

    voltage_ratio = vdc1 / (vdc2 + 1e-6)
    current_ratio = idc1 / (idc2 + 1e-6)
    power_ratio = power1 / (power2 + 1e-6)

    return pd.DataFrame([{
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
        "power_ratio": power_ratio,
    }])


def get_prediction(vdc1: float, vdc2: float, idc1: float, idc2: float, irr: float, pvt: float):
    """
    Returns (fault_key: str, confidence: float, all_probabilities: dict, features: pd.DataFrame).
    Raises RuntimeError if the model file isn't present yet.
    """
    if _model is None:
        raise RuntimeError(
            f"Model not found at {os.path.abspath(MODEL_PATH)}. "
            "Make sure solar_fault_model_engineered.pkl is in the models/ folder."
        )

    data = build_features(vdc1, vdc2, idc1, idc2, irr, pvt)
    prediction = _model.predict(data)[0]
    probabilities = _model.predict_proba(data)[0]
    confidence = float(max(probabilities))

    all_probabilities = {
        FAULT_NAMES[cls]: float(prob)
        for cls, prob in zip(_model.classes_, probabilities)
    }

    return FAULT_NAMES[prediction], confidence, all_probabilities, data
