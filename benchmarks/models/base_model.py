"""Abstract base class for OMR models.

All OMR models should inherit from BaseOMRModel to ensure
a consistent interface for benchmarking.
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseOMRModel(ABC):
    """Abstract base class for Optical Music Recognition models.

    All OMR models must implement the predict() method that takes
    a PIL Image and returns music notation in kern format (or similar).

    Args:
        name: Human-readable name for the model
        config: Optional configuration dict
    """

    def __init__(self, name: str, config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self._initialized = False

    def initialize(self):
        """Initialize the model (load weights, etc.).

        This is called automatically before the first prediction,
        but can be called manually for eager initialization.
        """
        if not self._initialized:
            logger.info(f"Initializing model: {self.name}")
            self._setup()
            self._initialized = True

    @abstractmethod
    def _setup(self):
        """Model-specific initialization logic.

        Override this method to load model weights, initialize
        neural networks, or perform other setup tasks.
        """
        pass

    @abstractmethod
    def _predict_impl(self, image: Image.Image, image_name: str = "image") -> str:
        """Core prediction logic.

        Args:
            image: Input image containing sheet music
            image_name: Name/identifier for the image (used for debug files)

        Returns:
            Music notation string (preferably in kern format)
        """
        pass

    def predict(self, image: Image.Image, image_name: str = "image") -> str:
        """Predict music notation from an image.

        Args:
            image: PIL Image containing sheet music
            image_name: Name/identifier for the image (used for debug files)

        Returns:
            Music notation string in kern format (or compatible format)

        Raises:
            ValueError: If image is invalid
            RuntimeError: If prediction fails
        """
        if image is None:
            raise ValueError("Image cannot be None")

        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL.Image, got {type(image)}")

        # Lazy initialization
        if not self._initialized:
            self.initialize()

        try:
            return self._predict_impl(image, image_name)
        except Exception as e:
            logger.error(f"Prediction failed for model {self.name}: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
