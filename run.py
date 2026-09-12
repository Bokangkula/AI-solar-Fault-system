"""
One-command launcher for SolarSense.

Run this instead of `streamlit run app.py` directly:

    python run.py

What it does, every time you run it:
1. Installs all required packages from requirements.txt
2. Checks if a Gemini API key is already saved in .env
   - If not, asks you to paste one in (just press Enter to skip and use
     the app without live AI explanations)
   - Saves it to .env so you're never asked again
3. Starts the Streamlit app in your browser

"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(REPO_ROOT, ".env")


def install_requirements():
    print("Installing dependencies (this can take a minute the first time)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        cwd=REPO_ROOT,
        check=True,
    )
    print("Dependencies installed.\n")


def get_existing_key():
    if not os.path.exists(ENV_PATH):
        return None
    with open(ENV_PATH, "r") as f:
        for line in f:
            if line.strip().startswith("GEMINI_API_KEY="):
                value = line.strip().split("=", 1)[1].strip()
                return value if value and value != "your_key_here" else None
    return None


def save_key(key: str):
    with open(ENV_PATH, "w") as f:
        f.write(f"GEMINI_API_KEY={key}\n")


def ensure_api_key():
    existing = get_existing_key()
    if existing:
        print("Found a saved Gemini API key in .env - using it.\n")
        return

    print("=" * 50)
    print("No Gemini API key found yet.")
    print("Get a free one at: https://aistudio.google.com/app/apikey")
    print("Paste it below, or just press Enter to skip for now")
    print("explanation instead of a live Gemini call).")
    print("=" * 50)

    key = input("Gemini API key: ").strip()

    if key:
        save_key(key)
        print("Saved to .env - you won't be asked again.\n")
    else:
        print("Skipped. You can add a key later by re-running this script.\n")


def run_app():
    print("Starting SolarSense...\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    install_requirements()
    ensure_api_key()
    run_app()
