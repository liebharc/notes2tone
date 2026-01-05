"""Wrapper for the homr (Homer's Optical Music Recognition) model.

homr is an end-to-end OMR system that combines oemer's UNet segmentation
with Polyphonic-TrOMR transformer to transcribe sheet music to MusicXML format.

Repository: https://github.com/liebharc/homr
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from PIL import Image
import logging

from .base_model import BaseOMRModel
from ..converters import convert_musicxml_to_kern

logger = logging.getLogger(__name__)


class HomrModel(BaseOMRModel):
    """Wrapper for the homr OMR model.

    This model uses the homr CLI to perform predictions.
    homr outputs MusicXML files, which are then converted to **kern format.

    Args:
        homr_path: Path to the homr executable (default: "homr")
        force_cpu: Force CPU mode even if GPU is available
        config: Additional configuration options

    Example:
        >>> model = HomrModel(homr_path="homr")
        >>> prediction = model.predict(image)
    """

    def __init__(
        self,
        homr_path: str = "homr",
        force_cpu: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__(name="homr", config=config)
        self.homr_path = homr_path
        self.force_cpu = force_cpu

    def _setup(self):
        """Verify that homr is installed and accessible."""
        import shutil

        # Check if homr executable exists in PATH
        homr_executable = shutil.which(self.homr_path)

        if homr_executable is None:
            raise RuntimeError(
                f"homr executable not found at: {self.homr_path}\n"
                "Install it with:\n"
                "  git clone https://github.com/liebharc/homr\n"
                "  cd homr\n"
                "  poetry install --only main,gpu  # for GPU\n"
                "  poetry install --only main      # for CPU\n"
                "Or specify the correct path with homr_path parameter"
            )

        logger.info(f"homr executable found at: {homr_executable}")

    def _predict_impl(self, image: Image.Image) -> str:
        """Run homr prediction on the image.

        Args:
            image: Input sheet music image

        Returns:
            Predicted notation as **kern string
        """
        # Create temporary directory for input and output
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            # Save input image
            input_path = tmp_dir_path / "input.png"
            image.save(input_path)

            # Build command
            cmd = [self.homr_path, str(input_path)]

            if self.force_cpu:
                cmd.append("--force-cpu")

            # Execute homr
            logger.debug(f"Running command: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=480,  # 8 minute timeout
                    cwd=str(tmp_dir_path),
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    raise RuntimeError(
                        f"homr failed with exit code {result.returncode}: {error_msg}"
                    )

                # homr outputs to <input_name>.musicxml in the same directory
                output_path = tmp_dir_path / "input.musicxml"

                if not output_path.exists():
                    raise RuntimeError(
                        f"homr did not produce expected output file: {output_path}"
                    )

                # Read MusicXML content
                with open(output_path, "r", encoding="utf-8") as f:
                    musicxml_content = f.read()

                # Debug: Save MusicXML for inspection
                debug_dir = Path("benchmarks/debug")
                debug_dir.mkdir(exist_ok=True)
                debug_path = (
                    debug_dir / f"homr_musicxml_{hash(musicxml_content) % 10000}.xml"
                )
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(musicxml_content)
                logger.info(f"Saved MusicXML debug file: {debug_path}")

                # Convert MusicXML to **kern format
                try:
                    kern_output = convert_musicxml_to_kern(musicxml_content)
                    logger.info(
                        f"Successfully converted MusicXML to **kern ({len(kern_output)} chars)"
                    )
                    return kern_output
                except Exception as e:
                    logger.error(f"Failed to convert MusicXML to **kern: {e}")
                    # Fall back to returning raw MusicXML if conversion fails
                    logger.warning("Returning raw MusicXML instead of **kern")
                    return musicxml_content

            except subprocess.TimeoutExpired:
                raise RuntimeError("homr prediction timed out (>8 minutes)")
