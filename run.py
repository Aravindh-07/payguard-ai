"""
run.py
------
Convenience CLI for setting up and running PayGuard AI end-to-end.

Usage:
    python run.py setup      # generate dataset + train model (first-time setup)
    python run.py generate   # regenerate the synthetic dataset only
    python run.py train      # (re)train the model only
    python run.py evaluate   # print evaluation metrics for the saved model
    python run.py demo       # run example transactions through the model
    python run.py test       # run the pytest suite
    python run.py dashboard  # launch the Streamlit dashboard
    python run.py api        # launch the FastAPI backend (uvicorn)
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def cmd_generate(_args):
    run([sys.executable, os.path.join("src", "data_generator.py")])


def cmd_train(_args):
    run([sys.executable, os.path.join("src", "train.py")])


def cmd_evaluate(_args):
    run([sys.executable, os.path.join("src", "evaluate.py")])


def cmd_demo(_args):
    run([sys.executable, os.path.join("src", "demo_transactions.py")])


def cmd_test(_args):
    run([sys.executable, "-m", "pytest", "tests/", "-v"])


def cmd_setup(args):
    print("=== PayGuard AI: full setup ===")
    cmd_generate(args)
    cmd_train(args)
    print("\nSetup complete. Try:  python run.py demo")


def cmd_dashboard(_args):
    run(["streamlit", "run", os.path.join("app", "dashboard.py")])


def cmd_api(_args):
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = os.environ.get("API_PORT", "8000")
    run(["uvicorn", "app.api:app", "--reload", "--host", host, "--port", str(port)])


def main():
    parser = argparse.ArgumentParser(description="PayGuard AI project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Generate dataset and train model (first-time setup)")
    sub.add_parser("generate", help="Regenerate the synthetic dataset")
    sub.add_parser("train", help="Train and save the fraud detection model")
    sub.add_parser("evaluate", help="Evaluate the saved model")
    sub.add_parser("demo", help="Run example transactions through the model")
    sub.add_parser("test", help="Run the automated test suite")
    sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    sub.add_parser("api", help="Launch the FastAPI backend")

    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
        "test": cmd_test,
        "dashboard": cmd_dashboard,
        "api": cmd_api,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
