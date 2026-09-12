# SolarSense

### *AI-powered fault triage for two-string PV solar systems*

---

## Overview

**SolarSense** analyzes measurements from a two-string PV solar system and
uses a trained machine learning model to detect the likely fault condition —
normal operation, short circuit, degradation, open circuit, or shadowing —
then explains the result in plain, technician-ready language using Gemini.

---

## Problem statement

Only around a quarter of Botswana's rural population has grid electricity
access, compared to near-universal access in cities, which is why solar PV
systems have become a key part of closing that gap in remote areas. But
once deployed, these systems are hard to maintain: the nearest qualified
technician is often a full day's travel away, and diagnosing a fault
typically requires an in-person visit with no way to know in advance what's
actually wrong. A fixable fault can mean weeks without power simply because
nobody nearby can identify it.

---

## Solution

SolarSense compares live readings from two PV strings — voltage, current,
irradiance, and temperature — and uses a Random Forest classifier trained
on engineered features (power, and the voltage/current/power differences
and ratios between the two strings) to detect the most likely fault. Instead
of a raw prediction, the result is turned into a clear, actionable
explanation for a technician, with a confidence score attached.

---

## How it works

1. A technician enters readings from both PV strings (voltage, current) plus
   irradiance and panel temperature
2. Engineered features are computed from the raw readings:
   power per string, and the difference/ratio between the two strings for
   voltage, current, and power
3. The trained Random Forest model classifies the fault
4. The confidence score is the model's maximum predicted class probability
5. Gemini (`gemini-3.5-flash-lite`) turns the raw prediction into a
   practical explanation covering what the condition means, possible
   causes, and what to inspect first — falling back to a general message if
   the API is unavailable, so the demo never breaks on stage

---

## Fault classes

| Class | Meaning |
|---|---|
| Normal operation | No fault detected |
| Short circuit | Fault between components — possible fire risk, inspect immediately |
| Degradation | Gradual output drop consistent with aging, not a sudden fault |
| Open circuit | Likely a broken or disconnected wire |
| Shadowing | One string is under more shade/dust/debris than the other |

---

## Project structure

```
AI-solar-Fault-system/
├── app.py                          # Streamlit UI - the interactive version of the CLI tool
├── requirements.txt
├── .env.example                    # Template for GEMINI_API_KEY
│
├── AI/
│   └── solarsense_ai.py            # Original CLI version (input()/print()-based)
│
├── utils/
│   ├── __init__.py
│   ├── predict.py                  # Shared prediction pipeline (used by both app.py and future scripts)
│   └── gemini.py                   # Gemini integration, shared with the same explanation logic as the CLI
│
├── models/
│   └── solar_fault_model_engineered.pkl   # Trained Random Forest model
│
└── Data/
    ├── dataset_amb.mat             # Ambient (irradiance/temperature) training data
    └── dataset_elec.mat            # Electrical (voltage/current) training data
```

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML model | scikit-learn (Random Forest) |
| Model storage | joblib |
| AI explanation | Gemini API (`google-genai`, model `gemini-3.5-flash-lite`) |
| Data processing | pandas |

---

## Running it locally

### 1. Clone the repository

```bash
git clone https://github.com/Deranged28/AI-solar-Fault-system.git
cd AI-solar-Fault-system
```

### 2. Run it

```bash
python run.py
```

That's it. This single command:
- Installs all dependencies automatically
- Asks for your Gemini API key the first time only (get a free one at
  https://aistudio.google.com/app/apikey), and saves it to `.env` so
  you're never asked again - or just press Enter to skip and use the app
  without live AI explanations
- Launches the app in your browser

 If you ever want to change or add your API key later, just run `python run.py` again.

### Running the original CLI version

```bash
python AI/solarsense_ai.py
```

(This requires `GEMINI_API_KEY` to already be set - run `python run.py`
at least once first so it's saved to `.env`.)

---

## Example input

**Readings:**

```
String 1 - voltage: 30V, current: 5A
String 2 - voltage: 30V, current: 5A
Irradiance: 800 W/m²
Temperature: 45°C
```

**Output includes:**

* Detected condition (e.g. Normal operation)
* Confidence score
* Plain-language, technician-facing explanation

---

## Future scope

* Direct sensor integration for automatic readings instead of manual entry
* Expanded fault taxonomy trained on additional real-world PV data
* Offline-friendly mode for technicians working without connectivity
* Fleet-level dashboards for rural electrification programs monitoring
  multiple installed systems at once

---

## AI tools disclosure

* Claude was used to build the Streamlit UI, the shared prediction
  pipeline, and to fix a path bug in the original CLI script
* Gemini (`gemini-3.5-flash-lite`) is used at runtime to generate
  technician-facing fault explanations

---
