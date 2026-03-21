"""Loader for the PRAIG/SMB (Sheet Music Benchmark) dataset.

The SMB dataset is designed for evaluating full-page OMR systems.
It contains printed music scores with ground truth annotations in kern format.

Reference: https://huggingface.co/datasets/PRAIG/SMB
"""

from typing import Iterator, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SMBDataset:
    """Wrapper for the PRAIG/SMB benchmark dataset.
    
    Args:
        split: Dataset split to load (default: "test")
        cache_dir: Optional directory to cache the dataset
        limit: Optional limit on number of samples to load
        token: HuggingFace API token for gated datasets (or set HF_TOKEN env var)
    """
    
    def __init__(
        self,
        split: str = "test",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
        token: Optional[str] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.limit = limit
        self.token = token
        self._dataset = None
        
    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace."""
        if self._dataset is None:
            try:
                from datasets import load_dataset
                logger.info(f"Loading PRAIG/SMB dataset (split: {self.split})...")
                self._dataset = load_dataset(
                    "PRAIG/SMB",
                    split=self.split,
                    cache_dir=self.cache_dir,
                    token=self.token  # Pass token for gated datasets
                )
                logger.info(f"Loaded {len(self._dataset)} samples")
            except ImportError:
                raise ImportError(
                    "The 'datasets' package is required. "
                    "Install it with: pip install datasets"
                )
            except Exception as e:
                error_msg = str(e)
                if "gated dataset" in error_msg.lower() or "authenticated" in error_msg.lower():
                    raise RuntimeError(
                        f"Failed to load SMB dataset: {e}\n\n"
                        "The PRAIG/SMB dataset is gated. To access it:\n"
                        "1. Request access at: https://huggingface.co/datasets/PRAIG/SMB\n"
                        "2. Login with: huggingface-cli login\n"
                        "   Or set HF_TOKEN environment variable\n"
                        "   Or pass token parameter to SMBDataset(token='your_token')"
                    )
                raise RuntimeError(f"Failed to load SMB dataset: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples.
        
        Yields:
            Dict with keys:
                - image: PIL.Image
                - ground_truth: str (kern notation)
                - metadata: dict (additional info like regions, symbols)
        """
        self._load_dataset()
        
        count = 0
        for item in self._dataset:
            if self.limit and count >= self.limit:
                break
                
            yield self._format_item(item)
            count += 1
    
    def __len__(self) -> int:
        """Return the number of samples (respecting limit)."""
        self._load_dataset()
        total = len(self._dataset)
        return min(total, self.limit) if self.limit else total
    
    def _format_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format a dataset item to a standardized structure.
        
        Args:
            item: Raw item from HuggingFace dataset
            
        Returns:
            Formatted item with image and ground_truth
        """
        # SMB dataset stores **kern in 'regions' array or 'page' object
        regions = item.get("regions", [])
        
        # Concatenate all region kern annotations
        kern_parts = []
        for region in regions:
            region_kern = region.get("kern", "")
            if region_kern:
                kern_parts.append(region_kern)
        
        # Join all regions (each region is a separate system/staff)
        ground_truth = "\n".join(kern_parts) if kern_parts else ""
        
        # Fallback: check page-level kern if regions are empty
        if not ground_truth:
            page = item.get("page", {})
            if isinstance(page, dict):
                ground_truth = page.get("kern", "")
        
        return {
            "image": item["image"],
            "ground_truth": ground_truth,
            "metadata": {
                "regions": regions,
                "width": item.get("original_width"),
                "height": item.get("original_height"),
            }
        }
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a specific sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Formatted sample dict
        """
        self._load_dataset()
        if index < 0 or index >= len(self._dataset):
            raise IndexError(f"Index {index} out of range [0, {len(self._dataset)})")
        
        return self._format_item(self._dataset[index])
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Support indexing with square brackets.
        
        Args:
            index: Sample index
            
        Returns:
            Formatted sample dict
        """
        return self.get_sample(index)
