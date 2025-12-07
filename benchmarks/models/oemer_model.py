"""Wrapper for the OeMeR (Optical Music Recognition) model.

OeMeR is an end-to-end OMR system that can transcribe sheet music
to MusicXML format.

Repository: https://github.com/BreezeWhite/oemer
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


class OemerModel(BaseOMRModel):
    """Wrapper for the OeMeR OMR model.
    
    This model uses the oemer CLI to perform predictions.
    OeMeR outputs MusicXML files, which are then read and returned.
    
    Args:
        oemer_path: Path to the oemer executable (default: "oemer")
        checkpoint_path: Optional path to custom model checkpoint
        use_tf: Use TensorFlow instead of ONNX runtime
        disable_deskew: Disable deskewing if images have no skew
        config: Additional configuration options
    
    Example:
        >>> model = OemerModel(oemer_path="/path/to/oemer")
        >>> prediction = model.predict(image)
    """
    
    def __init__(
        self,
        oemer_path: str = "oemer",
        checkpoint_path: Optional[str] = None,
        use_tf: bool = False,
        disable_deskew: bool = False,
        save_cache: bool = False,
        config: Optional[dict] = None
    ):
        super().__init__(name="OeMeR", config=config)
        self.oemer_path = oemer_path
        self.checkpoint_path = checkpoint_path
        self.use_tf = use_tf
        self.disable_deskew = disable_deskew
        self.save_cache = save_cache
        
    def _setup(self):
        """Verify that oemer is installed and accessible."""
        import shutil
        
        # Check if oemer executable exists in PATH
        oemer_executable = shutil.which(self.oemer_path)
        
        if oemer_executable is None:
            raise RuntimeError(
                f"oemer executable not found at: {self.oemer_path}\n"
                "Install it with: pip install oemer\n"
                "Or specify the correct path with oemer_path parameter"
            )
        
        logger.info(f"OeMeR executable found at: {oemer_executable}")
    
    def _predict_impl(self, image: Image.Image) -> str:
        """Run oemer prediction on the image.
        
        Args:
            image: Input sheet music image
            
        Returns:
            Predicted notation as MusicXML string
        """
        # Create temporary directory for input and output
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            
            # Save input image
            input_path = tmp_dir_path / "input.png"
            image.save(input_path)
            
            # Build command
            cmd = [self.oemer_path, str(input_path), "-o", str(tmp_dir_path)]
            
            if self.use_tf:
                cmd.append("--use-tf")
            
            if self.disable_deskew:
                cmd.append("--without-deskew")
            
            if self.save_cache:
                cmd.append("--save-cache")
            
            if self.checkpoint_path:
                cmd.extend(["--checkpoint", self.checkpoint_path])
            
            # Execute oemer
            logger.debug(f"Running command: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=480  # 8 minute timeout
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    raise RuntimeError(
                        f"oemer failed with exit code {result.returncode}: {error_msg}"
                    )
                
                # OeMeR outputs to <input_name>.musicxml
                output_path = tmp_dir_path / "input.musicxml"
                
                if not output_path.exists():
                    raise RuntimeError(
                        f"OeMeR did not produce expected output file: {output_path}"
                    )
                
                # Read MusicXML content
                with open(output_path, 'r', encoding='utf-8') as f:
                    musicxml_content = f.read()
                
                # Debug: Save MusicXML for inspection
                debug_dir = Path("benchmarks/debug")
                debug_dir.mkdir(exist_ok=True)
                debug_path = debug_dir / f"musicxml_{hash(musicxml_content) % 10000}.xml"
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(musicxml_content)
                logger.info(f"Saved MusicXML debug file: {debug_path}")
                
                # Convert MusicXML to **kern format
                try:
                    kern_output = convert_musicxml_to_kern(musicxml_content)
                    logger.info(f"Successfully converted MusicXML to **kern ({len(kern_output)} chars)")
                    return kern_output
                except Exception as e:
                    logger.error(f"Failed to convert MusicXML to **kern: {e}")
                    # Fall back to returning raw MusicXML if conversion fails
                    logger.warning("Returning raw MusicXML instead of **kern")
                    return musicxml_content
                    
            except subprocess.TimeoutExpired:
                raise RuntimeError("oemer prediction timed out (>8 minutes)")
