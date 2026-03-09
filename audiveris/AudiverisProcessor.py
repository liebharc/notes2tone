import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Maximum command length to prevent shell errors (Windows typically has ~8000 char limit)
MAX_COMMAND_LENGTH = 7000


class AudiverisProcessor:
    """Process images with Audiveris in batches, respecting command length limits."""

    def __init__(
        self,
        audiveris_path: str,
        output_dir: str,
        upscale_factor: float = 2.0,
        upscale_max_side_threshold: int = 3500,
    ):
        """
        Args:
            audiveris_path: Path to Audiveris executable
            output_dir: Directory where Audiveris will output files
            upscale_factor: Scale factor applied before Audiveris (1.0 disables upscaling)
            upscale_max_side_threshold: Only upscale images whose max side is below this threshold
        """
        self.audiveris_path = audiveris_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if upscale_factor < 1.0:
            raise ValueError("upscale_factor must be >= 1.0")
        self.upscale_factor = upscale_factor
        self.upscale_max_side_threshold = upscale_max_side_threshold

    def _save_image_to_temp(self, image, sample_id: str) -> str:
        """Save a PIL image to a temporary file (optionally upscaled) and return its path."""
        from PIL import Image

        temp_dir = Path(tempfile.gettempdir()) / "audiveris_batch"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = temp_dir / f"{sample_id}.png"

        if isinstance(image, Image.Image):
            source_img = image
            close_after = False
        else:
            source_img = Image.open(str(image))
            close_after = True

        out_img = source_img
        try:
            width, height = out_img.size
            should_upscale = self.upscale_factor > 1.0 and max(width, height) < self.upscale_max_side_threshold

            if should_upscale:
                new_size = (max(1, int(width * self.upscale_factor)), max(1, int(height * self.upscale_factor)))
                out_img = out_img.resize(new_size, resample=Image.Resampling.LANCZOS)
                logger.info(f"Upscaled {sample_id}: {width}x{height} -> {new_size[0]}x{new_size[1]}")

            out_img.save(str(temp_path))
        finally:
            if close_after:
                source_img.close()
            if out_img is not source_img:
                out_img.close()

        return str(temp_path)

    def _build_command_args(self, image_paths: List[str]) -> List[str]:
        """Build Audiveris command args for subprocess.run without shell=True."""
        return [
            self.audiveris_path,
            "-batch",
            "-export",
            "-output",
            str(self.output_dir),
            "--",
            *image_paths,
        ]

    def _command_length(self, args: List[str]) -> int:
        # Approximate command line size with spaces between args.
        return sum(len(p) for p in args) + max(len(args) - 1, 0)

    def _execute_batch(self, image_paths: List[str]) -> bool:
        """Execute one Audiveris batch and wait for completion."""
        args = self._build_command_args(image_paths)
        try:
            logger.info(f"Executing Audiveris batch with {len(image_paths)} images")
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=3600,
                shell=False,
            )
            if result.returncode == 0:
                logger.info("Audiveris batch completed successfully")
                return True
            logger.error(f"Audiveris failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Audiveris command timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error executing Audiveris command: {e}")
            return False

    def process_dataset(self, dataset, limit: int = None) -> List[Dict[str, Any]]:
        """
        Process all images from the dataset with Audiveris.

        Args:
            dataset: SMBDataset instance
            limit: Optional limit on number of images to process
        """
        image_paths_batch: List[str] = []
        processed_samples: List[Dict[str, Any]] = []
        batch_count = 0

        for idx, sample in enumerate(dataset):
            if limit and idx >= limit:
                break

            sample_id = sample.get("sample_id", f"sample_{idx:06d}")
            temp_image_path = self._save_image_to_temp(sample["image"], sample_id)
            sample["audiveris_input_path"] = temp_image_path
            processed_samples.append(sample)

            candidate_batch = image_paths_batch + [temp_image_path]
            if image_paths_batch and self._command_length(self._build_command_args(candidate_batch)) > MAX_COMMAND_LENGTH:
                success = self._execute_batch(image_paths_batch)
                batch_count += 1
                if success:
                    logger.info(f"Batch {batch_count} processed ({len(image_paths_batch)} images)")
                else:
                    logger.warning(f"Batch {batch_count} failed, continuing")
                image_paths_batch = [temp_image_path]
            else:
                image_paths_batch = candidate_batch

        if image_paths_batch:
            success = self._execute_batch(image_paths_batch)
            batch_count += 1
            if success:
                logger.info(f"Final batch {batch_count} processed ({len(image_paths_batch)} images)")
            else:
                logger.warning(f"Final batch {batch_count} failed")

        logger.info(f"Dataset processing complete. Processed {batch_count} batches.")
        return processed_samples

