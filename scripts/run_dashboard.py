"""CLI launcher for the Milestone 3 Streamlit Executive Dashboard."""

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Streamlit Employee Motion Executive Dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve Streamlit dashboard on.")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode without auto-opening a browser.")
    args = parser.parse_args()

    app_path = PROJECT_ROOT / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    print(f"Launching Employee Motion Executive Dashboard on http://localhost:{args.port} ...")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
