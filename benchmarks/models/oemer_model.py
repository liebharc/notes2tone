"""Wrapper for the OeMeR (Optical Music Recognition) model.

OeMeR is an end-to-end OMR system that can transcribe sheet music
to MusicXML format.

Repository: https://github.com/BreezeWhite/oemer
"""

import tempfile
from pathlib import Path
from typing import Optional
from PIL import Image
import logging
from argparse import Namespace

from .base_model import BaseOMRModel
from ..converters import convert_musicxml_to_kern

logger = logging.getLogger(__name__)


class OemerModel(BaseOMRModel):
    """Wrapper for the OeMeR OMR model.

    This model calls OEMER directly as a Python module.
    OeMeR outputs MusicXML files, which are then converted to **kern format.

    Args:
        oemer_module_path: Optional path to OEMER module directory (e.g., "/home/jovyan/work/oemer").
                          Only needed if OEMER is not in PYTHONPATH. Will be added to sys.path.
        checkpoint_path: Optional path to custom model checkpoint (currently unused)
        use_tf: Use TensorFlow instead of ONNX runtime
        disable_deskew: Disable deskewing if images have no skew
        save_cache: Save intermediate predictions for faster re-runs
        config: Additional configuration options

    Example:
        >>> # If OEMER installed via pip
        >>> model = OemerModel()
        >>>
        >>> # If OEMER in custom location
        >>> model = OemerModel(oemer_module_path="/path/to/oemer")
        >>>
        >>> prediction = model.predict(image)
    """

    def __init__(
        self,
        oemer_module_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        use_tf: bool = False,
        disable_deskew: bool = False,
        save_cache: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__(name="OeMeR", config=config)
        self.oemer_module_path = oemer_module_path
        self.checkpoint_path = checkpoint_path
        self.use_tf = use_tf
        self.disable_deskew = disable_deskew
        self.save_cache = save_cache

    def _setup(self):
        """Verify that oemer is installed and accessible."""
        import sys

        # Add custom OEMER path to sys.path if provided
        if self.oemer_module_path:
            oemer_path = str(Path(self.oemer_module_path).resolve())
            if oemer_path not in sys.path:
                sys.path.insert(0, oemer_path)
                logger.info(f"Added OEMER module path to sys.path: {oemer_path}")

        # Try to import OEMER
        try:
            from oemer.ete import extract
            import oemer

            logger.info(f"OEMER module loaded successfully from: {oemer.__file__}")
        except ImportError as e:
            raise RuntimeError(
                f"OEMER module not found: {e}\n"
                "Install it with: pip install -e /path/to/oemer\n"
                "Or: pip install oemer\n"
                f"Or specify oemer_module_path parameter"
            )

    def _predict_impl(self, image: Image.Image) -> str:
        """Run oemer prediction on the image.

        Args:
            image: Input sheet music image

        Returns:
            Predicted notation as **kern format
        """
        from oemer.ete import extract, clear_data

        # Clear any previous data from OEMER's internal layers
        clear_data()

        # Create temporary directory for input and output
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            # Save input image
            input_path = tmp_dir_path / "input.png"
            image.save(input_path)

            # Create args namespace (like argparse would)
            args = Namespace(
                img_path=str(input_path),
                output_path=str(tmp_dir_path),
                use_tf=self.use_tf,
                save_cache=self.save_cache,
                without_deskew=self.disable_deskew,
            )

            # Call OEMER directly (not via CLI)
            logger.info(f"Running OEMER prediction on {input_path}")
            try:
                musicxml_path = extract(args)
                logger.info(f"OEMER finished, output: {musicxml_path}")

                # Read MusicXML content
                with open(musicxml_path, "r", encoding="utf-8") as f:
                    musicxml_content = f.read()

                # Debug: Save MusicXML for inspection
                debug_dir = Path("benchmarks/debug")
                debug_dir.mkdir(exist_ok=True)
                debug_path = (
                    debug_dir / f"musicxml_{hash(musicxml_content) % 10000}.xml"
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
                    logger.warning("Returning raw MusicXML instead of **kern")
                    return musicxml_content

            except Exception as e:
                raise RuntimeError(f"OEMER prediction failed: {e}")
