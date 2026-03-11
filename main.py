from __future__ import annotations

import argparse
from pathlib import Path

from musicdiff import diff_ml_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run musicdiff ML training evaluation for GT and predicted score folders."
    )
    parser.add_argument(
        "--gt",
        default="generated/gt",
        help="Path to the ground-truth folder (default: generated/gt).",
    )
    parser.add_argument(
        "--pred",
        default="generated/xml_scaled",
        help="Path to the predicted folder (default: generated/pred).",
    )
    parser.add_argument(
        "--out",
        default="generated",
        help="Output folder for output.csv (default: generated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_score, output_csv = diff_ml_training(
        predicted_folder=args.pred,
        ground_truth_folder=args.gt,
        output_folder=str(out_dir),
    )

    print(f"ML training overall score: {overall_score}")
    print(f"CSV written to: {output_csv}")


if __name__ == "__main__":
    main()
