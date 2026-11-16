"""Standalone CLI generator for Milestone 3 Executive Reports (CSV, JSON, PNG, PDF)."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # noqa: F401
from src.config import REPORTS_OUTPUT_DIR, TELEMETRY_OUTPUT_DIR
from src.dashboard.reporter import generate_executive_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Milestone 3 Executive KPI Reports.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(TELEMETRY_OUTPUT_DIR / "temporal_features.csv"),
        help="Input classified telemetry CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_OUTPUT_DIR),
        help="Target directory for generated reports.",
    )
    args = parser.parse_args()

    outputs = generate_executive_reports(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
    )

    print("=================================================================")
    print("  Milestone 3: Executive Reports Generated Successfully")
    for k, v in outputs.items():
        print(f"  - {k}: {v}")
    print("=================================================================")


if __name__ == "__main__":
    main()
