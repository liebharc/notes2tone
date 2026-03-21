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

logger = logging.getLogger(__name__)


class HomrModel(BaseOMRModel):
    """Wrapper for the homr OMR model.

    This model uses the homr CLI to perform predictions.
    homr outputs MusicXML files.

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
        homr_dir: Optional[str] = None,
        force_cpu: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__(name="homr", config=config)
        self.homr_path = homr_path
        self.homr_dir = homr_dir
        self.force_cpu = force_cpu
        self.homr_executable = None  # Will be set in _setup()

    def _setup(self):
        """Verify that homr is installed and accessible."""
        import shutil

        # If homr_dir is set, we're using poetry run homr
        if self.homr_dir:
            homr_dir_path = Path(self.homr_dir)
            if not homr_dir_path.exists():
                raise RuntimeError(f"HOMR directory not found: {self.homr_dir}")
            if not (homr_dir_path / "pyproject.toml").exists():
                raise RuntimeError(
                    f"Not a valid HOMR repo (missing pyproject.toml): {self.homr_dir}"
                )
            logger.info(f"Using HOMR from: {self.homr_dir} (poetry run)")
            return

        # Check if homr executable exists in PATH
        self.homr_executable = shutil.which(self.homr_path)

        if self.homr_executable is None:
            raise RuntimeError(
                f"homr executable not found at: {self.homr_path}\n"
                "Install it with:\n"
                "  git clone https://github.com/liebharc/homr\n"
                "  cd homr\n"
                "  poetry install --only main,gpu  # for GPU\n"
                "  poetry install --only main      # for CPU\n"
                "Or specify the correct path with homr_path parameter"
            )

        logger.info(f"homr executable found at: {self.homr_executable}")

    def _predict_impl(self, image: Image.Image, image_name: str = "image") -> str:
        """Run homr prediction on the image.

        Args:
            image: Input sheet music image
            image_name: Name/identifier for the image

        Returns:
            Predicted notation as MusicXML string
        """
        # Create temporary directory for input and output
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            # Save input image
            input_path = tmp_dir_path / "input.png"
            image.save(input_path)

            # Build command based on whether we're using poetry run
            if self.homr_dir:
                # Use homr executable directly from the homr venv
                homr_executable = Path(self.homr_dir) / ".venv" / "bin" / "homr"

                if not homr_executable.exists():
                    raise RuntimeError(
                        f"HOMR executable not found at: {homr_executable}\n"
                        f"Install HOMR with:\n"
                        f"  cd {self.homr_dir}\n"
                        f"  poetry install --only main,gpu"
                    )

                cmd = [str(homr_executable), str(input_path)]
                if self.force_cpu:
                    cmd.append("--force-cpu")
                run_cwd = str(tmp_dir_path)
                use_shell = False
            else:
                # Use direct homr command (from stored executable path)
                cmd = [self.homr_executable, str(input_path)]
                if self.force_cpu:
                    cmd.append("--force-cpu")
                run_cwd = str(tmp_dir_path)
                # On Windows, CMD files need shell=True to execute properly
                import sys

                use_shell = sys.platform == "win32"

            # Execute homr
            logger.debug(f"Running command: {' '.join(cmd)} (cwd: {run_cwd})")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=480,  # 8 minute timeout
                    cwd=run_cwd,
                    shell=use_shell,
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

                # Save MusicXML output
                xml_output_dir = Path("benchmarks/output/homr/xml")
                xml_output_dir.mkdir(parents=True, exist_ok=True)
                xml_path = xml_output_dir / f"{image_name}.xml"
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(musicxml_content)
                logger.info(f"Saved MusicXML output: {xml_path}")

                return musicxml_content

            except subprocess.TimeoutExpired:
                raise RuntimeError("homr prediction timed out (>8 minutes)")
