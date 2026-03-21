"""Wrapper for the homr (Homer's Optical Music Recognition) model.

homr is an end-to-end OMR system that combines oemer's UNet segmentation
with Polyphonic-TrOMR transformer to transcribe sheet music to MusicXML format.

Repository: https://github.com/liebharc/homr
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import shutil
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
        self.use_poetry = False

    def _detect_local_homr_repo(self) -> Optional[Path]:
        """Try to find a sibling HOMR repo next to the notes2tone root."""
        # .../notes2tone/benchmarks/models/homr_model.py -> .../GIT_NEU/homr
        candidate = Path(__file__).resolve().parents[3] / "homr"
        if candidate.exists() and (candidate / "pyproject.toml").exists():
            return candidate
        return None

    def _find_homr_in_repo_venv(self, repo_dir: Path) -> Optional[Path]:
        """Find HOMR executable inside a repository-local virtual environment."""
        candidates = [
            repo_dir / ".venv" / "Scripts" / "homr.exe",
            repo_dir / ".venv" / "Scripts" / "homr.cmd",
            repo_dir / ".venv" / "Scripts" / "homr",
            repo_dir / ".venv" / "bin" / "homr",
        ]
        return next((p for p in candidates if p.exists()), None)

    def _setup(self):
        """Verify that homr is installed and accessible."""
        # If homr_dir is set, resolve executable from repo venv or poetry run.
        if self.homr_dir:
            homr_dir_path = Path(self.homr_dir)
            if not homr_dir_path.exists():
                raise RuntimeError(f"HOMR directory not found: {self.homr_dir}")
            if not (homr_dir_path / "pyproject.toml").exists():
                raise RuntimeError(
                    f"Not a valid HOMR repo (missing pyproject.toml): {self.homr_dir}"
                )

            repo_homr = self._find_homr_in_repo_venv(homr_dir_path)
            if repo_homr is not None:
                self.homr_executable = str(repo_homr)
                self.use_poetry = False
                logger.info(f"Using HOMR executable: {self.homr_executable}")
                return

            if shutil.which("poetry"):
                self.use_poetry = True
                logger.info(f"Using HOMR from: {self.homr_dir} (poetry run)")
                return

            raise RuntimeError(
                f"No HOMR executable found in {self.homr_dir}/.venv and poetry is not in PATH.\n"
                "Install HOMR in its repo environment or install poetry."
            )

        # Check if homr executable exists in PATH first.
        self.homr_executable = shutil.which(self.homr_path)

        if self.homr_executable is not None:
            logger.info(f"homr executable found at: {self.homr_executable}")
            return

        # Fallback: auto-detect sibling homr repository.
        local_repo = self._detect_local_homr_repo()
        if local_repo is not None:
            self.homr_dir = str(local_repo)
            logger.info(f"Auto-detected HOMR repo: {self.homr_dir}")
            self._setup()
            return

        raise RuntimeError(
            f"homr executable not found at: {self.homr_path}\n"
            "Install it with:\n"
            "  git clone https://github.com/liebharc/homr\n"
            "  cd homr\n"
            "  poetry install --only main,gpu  # for GPU\n"
            "  poetry install --only main      # for CPU\n"
            "Or specify the correct path with homr_path parameter"
        )

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

            # Build command based on whether we're using repo-local HOMR or PATH.
            if self.homr_dir:
                if self.use_poetry:
                    cmd = ["poetry", "run", "homr", str(input_path)]
                    run_cwd = str(Path(self.homr_dir))
                elif self.homr_executable:
                    cmd = [self.homr_executable, str(input_path)]
                    run_cwd = str(tmp_dir_path)
                else:
                    raise RuntimeError(
                        "HOMR setup is incomplete: neither executable nor poetry mode is available."
                    )

                if self.force_cpu:
                    cmd.append("--force-cpu")

                # On Windows, .cmd entry points may require shell=True.
                import sys

                use_shell = sys.platform == "win32"
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
