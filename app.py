import streamlit as st

from utils.predict import get_prediction, _model
from utils.gemini import get_explanation

st.set_page_config(page_title="SolarSense", page_icon="\u2600\ufe0f", layout="centered")

FAULT_INFO = {
    "normal": {
        "label": "Normal operation",
        "advice": "No fault detected. The system is operating normally.",
        "severity": "success",
    },
    "short_circuit": {
        "label": "Short circuit",
        "advice": "A short circuit was detected between components. This can be a fire "
        "risk - a technician should inspect wiring and connectors before "
        "the system is used further.",
        "severity": "danger",
    },
    "degradation": {
        "label": "Degradation",
        "advice": "Output has dropped in a way consistent with panel or component "
        "aging rather than a sudden fault. Monitor performance over time and "
        "plan for possible replacement.",
        "severity": "warning",
    },
    "open_circuit": {
        "label": "Open circuit",
        "advice": "A broken or disconnected wire is likely somewhere in the circuit. "
        "Check all cable connections between strings, controller, and battery.",
        "severity": "danger",
    },
    "shadowing": {
        "label": "Shadowing",
        "advice": "One string appears to be partially shaded compared to the other. "
        "Check for dust, debris, or shadows (e.g. from trees) on the panel.",
        "severity": "warning",
    },
}


def render_result(fault_key: str, confidence: float, all_probabilities: dict, features, readings: dict, use_ai_explanation: bool):
    info = FAULT_INFO[fault_key]
    box = {"success": st.success, "warning": st.warning, "danger": st.error}[info["severity"]]

    box(f"**{info['label']}** \u2014 {confidence * 100:.1f}% confidence")

    st.subheader("Model output")
    st.caption("Predicted probability for every class the model considered - this is the model's real output, not a canned message.")
    chart_data = {FAULT_INFO[k]["label"]: v for k, v in all_probabilities.items()}
    st.bar_chart(chart_data)

    with st.expander("See the engineered features fed into the model"):
        st.dataframe(features.T.rename(columns={0: "value"}))

    st.subheader("What to do")
    if use_ai_explanation:
        with st.spinner("Asking Gemini for a technician-level explanation..."):
           ai_text = get_explanation(info["label"], confidence, readings)
        st.write(ai_text)
    else:
        st.write(info["advice"])

    st.caption(
        "This is a preliminary ML-based triage result, not a certified diagnosis. "
        "A qualified technician should confirm before repair work."
    )


def main():
    st.title("\u2600\ufe0f SolarSense")
    st.caption("AI-powered fault triage for two-string PV systems")

    with st.sidebar:
        st.header("Settings")
        st.success("SolarSense AI is ready", icon="✅")

    if _model is None:
        st.error(
            "Model file not found at `models/solar_fault_model_engineered.pkl`. "
            "Make sure it's present in the repo before running predictions.",
            icon="\u26a0\ufe0f",
        )

    st.header("1. Enter readings")
    st.caption("Enter live measurements from both PV strings.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**String 1**")
        vdc1 = st.number_input("DC voltage - string 1 (V)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        idc1 = st.number_input("DC current - string 1 (A)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    with col2:
        st.markdown("**String 2**")
        vdc2 = st.number_input("DC voltage - string 2 (V)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        idc2 = st.number_input("DC current - string 2 (A)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

    col3, col4 = st.columns(2)
    with col3:
        irr = st.number_input("Irradiance (W/m\u00b2)", min_value=0.0, max_value=1500.0, value=800.0, step=10.0)
    with col4:
        pvt = st.number_input("Panel temperature (\u00b0C)", min_value=-10.0, max_value=90.0, value=45.0, step=0.5)

    use_ai_explanation = st.checkbox("Use Gemini for a technician-level explanation", value=False)

    st.divider()

    if st.button("Run diagnosis", type="primary", disabled=(_model is None)):
        readings = {"vdc1": vdc1, "vdc2": vdc2, "idc1": idc1, "idc2": idc2, "irr": irr, "pvt": pvt}
        fault_key, confidence, all_probabilities, features = get_prediction(**readings)

        st.header("2. Diagnosis")
        render_result(fault_key, confidence, all_probabilities, features, readings, use_ai_explanation)


if __name__ == "__main__":
    main()
