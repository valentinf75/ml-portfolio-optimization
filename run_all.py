# run_all.py

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(script_name: str, description: str):
    "Run a single script from the scripts/ folder."
    # print("STEP: {description}")
    # print("Running: {script_name}")

    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run([sys.executable, str(script_path)], check=True)

    #if result.returncode == 0:
    #    print(" {description} completed successfully")
    #else:
    #    print(" {description} finished with return code {result.returncode}")


def main():
    #Data preparation
    run_step("run_prepare.py", "Prepare technical features")

    #Baseline strategies (Equal-Weight and Markowitz)
    run_step("run_baselines.py", "Run classical baselines")

    #Supervised learning baseline (Random Forest)
    run_step("run_random_forest.py", "Run Random Forest supervised strategy")
    #Supervised learning baseline (Logistic Regression)
    run_step("run_logistic_regression.py", "Run Logistic Regression supervised strategy")


if __name__ == "__main__":
    main()