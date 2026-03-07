"""Main benchmark runner for evaluating OMR models.

This script provides a command-line interface for running
benchmarks on different OMR models using standardized datasets.
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from benchmarks.datasets import SMBDataset
from benchmarks.models.base_model import BaseOMRModel
from benchmarks.models.oemer_model import OemerModel
from benchmarks.models.homr_model import HomrModel
from benchmarks.eval import calculate_metrics, format_metrics

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates the evaluation of OMR models on benchmark datasets.

    Args:
        dataset: Dataset to evaluate on
        output_dir: Directory to save results
        save_predictions: Whether to save individual predictions
    """

    def __init__(
        self,
        dataset: SMBDataset,
        output_dir: Path = Path("benchmarks/results"),
        save_predictions: bool = False,
    ):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_predictions = save_predictions

    def evaluate_model(self, model: BaseOMRModel) -> dict[str, any]:
        """Evaluate a single model on the dataset.

        Args:
            model: OMR model to evaluate

        Returns:
            Dictionary with evaluation results and metrics
        """
        logger.info(f"Evaluating model: {model.name}")

        predictions = []
        ground_truths = []
        errors = []

        # Initialize model
        model.initialize()

        # Evaluate on each sample
        dataset_size = len(self.dataset)
        for idx, item in enumerate(tqdm(self.dataset, desc=f"Evaluating {model.name}")):
            try:
                image = item["image"]
                ground_truth = item["ground_truth"]

                # Get image name/identifier
                image_name = item.get("sample_id", item.get("filename", f"image{idx}"))
                logger.info(f"Processing image: {image_name}")

                # Get prediction
                prediction = model.predict(image, image_name)

                predictions.append(prediction)
                ground_truths.append(ground_truth)

            except Exception as e:
                logger.error(f"Error on sample {idx}: {e}")
                errors.append({"sample_index": idx, "error": str(e)})
                # Add empty prediction to maintain alignment
                predictions.append("")
                ground_truths.append(item.get("ground_truth", ""))

        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)

        # Add error information
        metrics["num_errors"] = len(errors)
        metrics["error_rate"] = len(errors) / dataset_size if dataset_size > 0 else 0.0

        results = {
            "model_name": model.name,
            "model_config": model.config,
            "metrics": metrics,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }

        # Optionally save predictions
        if self.save_predictions:
            results["predictions"] = [
                {"prediction": pred, "ground_truth": gt}
                for pred, gt in zip(predictions, ground_truths)
            ]

        return results

    def save_results(self, results: dict[str, any], filename: str):
        """Save evaluation results to JSON file.

        Args:
            results: Results dictionary from evaluate_model()
            filename: Output filename (without path)
        """
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")

    def compare_models(self, results_list: list[dict[str, any]]) -> str:
        """Generate a comparison table for multiple models.

        Args:
            results_list: List of result dictionaries

        Returns:
            Formatted comparison string
        """
        if not results_list:
            return "No results to compare"

        lines = ["\n" + "=" * 80, "Model Comparison", "=" * 80, ""]

        # Header
        lines.append(f"{'Model':<20} {'Mean NED':<12} {'Perfect':<12} {'Errors':<12}")
        lines.append("-" * 80)

        # Sort by mean NED (lower is better)
        sorted_results = sorted(results_list, key=lambda x: x["metrics"]["mean_ned"])

        # Add each model
        for result in sorted_results:
            name = result["model_name"]
            metrics = result["metrics"]

            mean_ned = f"{metrics['mean_ned']:.4f}"
            perfect = f"{metrics['perfect_matches']}/{metrics['total_samples']}"
            errors = f"{metrics['num_errors']}"

            lines.append(f"{name:<20} {mean_ned:<12} {perfect:<12} {errors:<12}")

        lines.append("=" * 80 + "\n")

        return "\n".join(lines)


def get_available_models() -> dict[str, BaseOMRModel]:
    """Get dictionary of available models.

    Add new models here to include them in benchmarks.

    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        "oemer": OemerModel(oemer_module_path="oemer"),
        "oemer-tf": OemerModel(
            oemer_module_path="oemer", use_tf=True
        ),  # Use TensorFlow (GPU)
        "homr": HomrModel(homr_path="homr"),
        # Add more models here:
        # "audiveris": AudiverisModel(),
    }


def main():
    """Main entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark OMR models on standardized datasets"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to evaluate (default: all available)",
        choices=list(get_available_models().keys()) + ["all"],
    )

    parser.add_argument(
        "--dataset",
        default="smb",
        choices=["smb"],
        help="Dataset to use for evaluation (default: smb)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for testing)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory to save results",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions to results file",
    )

    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only show comparison (requires existing result files)",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated datasets (or set HF_TOKEN in .env file)",
    )

    args = parser.parse_args()

    # Load dataset with token from args or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if args.dataset == "smb":
        dataset = SMBDataset(limit=args.limit, token=hf_token)
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    # Get models to evaluate
    available_models = get_available_models()

    if args.models and "all" not in args.models:
        models_to_eval = {
            name: model
            for name, model in available_models.items()
            if name in args.models
        }
    else:
        models_to_eval = available_models

    if not models_to_eval:
        logger.error("No models selected for evaluation")
        sys.exit(1)

    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset=dataset,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
    )

    # Evaluate each model
    all_results = []

    for model_name, model in models_to_eval.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting evaluation: {model_name}")
        logger.info(f"{'=' * 60}\n")

        try:
            results = runner.evaluate_model(model)
            all_results.append(results)

            # Save individual results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{args.dataset}_{timestamp}.json"
            runner.save_results(results, filename)

            # Print metrics
            print(format_metrics(results["metrics"], model_name))

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}", exc_info=True)

    # Print comparison
    if len(all_results) > 1:
        print(runner.compare_models(all_results))

    # Save combined results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"comparison_{args.dataset}_{timestamp}.json"
        runner.save_results(
            {
                "dataset": args.dataset,
                "models": all_results,
                "timestamp": datetime.now().isoformat(),
            },
            combined_filename,
        )

    logger.info("\nBenchmark completed!")


if __name__ == "__main__":
    main()
