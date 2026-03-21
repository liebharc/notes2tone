"""Metrics for evaluating OMR system performance.

The primary metric is OMR-NED (OMR Normalized Edit Distance),
which is the Levenshtein distance normalized by the length of
the ground truth sequence.

Reference: PRAIG/SMB benchmark paper
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def levenshtein_distance(seq1: str, seq2: str) -> int:
    """Calculate Levenshtein (edit) distance between two sequences.
    
    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) needed to transform
    one string into another.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Edit distance (integer >= 0)
    
    Example:
        >>> levenshtein_distance("kitten", "sitting")
        3
    """
    len1, len2 = len(seq1), len(seq2)
    
    # Create DP table
    dp = np.zeros((len1 + 1, len2 + 1), dtype=np.int32)
    
    # Initialize base cases
    for i in range(len1 + 1):
        dp[i, 0] = i
    for j in range(len2 + 1):
        dp[0, j] = j
    
    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dp[i, j] = min(
                dp[i-1, j] + 1,      # deletion
                dp[i, j-1] + 1,      # insertion
                dp[i-1, j-1] + cost  # substitution
            )
    
    return int(dp[len1, len2])


def omr_ned(prediction: str, ground_truth: str) -> float:
    """Calculate OMR Normalized Edit Distance.
    
    OMR-NED is the Levenshtein distance divided by the length of
    the ground truth. Lower scores are better, with 0.0 being perfect.
    
    Args:
        prediction: Predicted notation string
        ground_truth: Ground truth notation string
        
    Returns:
        Normalized edit distance (float in [0, inf), typically [0, 2])
        
    Example:
        >>> omr_ned("abcd", "abce")
        0.25
    """
    if not ground_truth:
        # If ground truth is empty, return 1.0 if prediction is non-empty
        return 1.0 if prediction else 0.0
    
    edit_dist = levenshtein_distance(prediction, ground_truth)
    normalized = edit_dist / len(ground_truth)
    
    return normalized


def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Calculate comprehensive metrics for a set of predictions.
    
    Args:
        predictions: List of predicted notation strings
        ground_truths: List of ground truth notation strings
        
    Returns:
        Dictionary with metrics:
            - mean_ned: Average OMR-NED across all samples
            - median_ned: Median OMR-NED
            - std_ned: Standard deviation of OMR-NED
            - min_ned: Best (minimum) OMR-NED
            - max_ned: Worst (maximum) OMR-NED
            - perfect_matches: Number of perfect predictions (NED = 0)
            - total_samples: Total number of samples evaluated
    
    Raises:
        ValueError: If predictions and ground_truths have different lengths
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truths "
            f"({len(ground_truths)}) must have the same length"
        )
    
    if not predictions:
        return {
            "mean_ned": 0.0,
            "median_ned": 0.0,
            "std_ned": 0.0,
            "min_ned": 0.0,
            "max_ned": 0.0,
            "perfect_matches": 0,
            "total_samples": 0
        }
    
    # Calculate NED for each sample
    ned_scores = [
        omr_ned(pred, gt)
        for pred, gt in zip(predictions, ground_truths)
    ]
    
    ned_array = np.array(ned_scores)
    
    return {
        "mean_ned": float(np.mean(ned_array)),
        "median_ned": float(np.median(ned_array)),
        "std_ned": float(np.std(ned_array)),
        "min_ned": float(np.min(ned_array)),
        "max_ned": float(np.max(ned_array)),
        "perfect_matches": int(np.sum(ned_array == 0)),
        "total_samples": len(ned_scores)
    }


def format_metrics(metrics: Dict[str, float], model_name: str = "Model") -> str:
    """Format metrics as a human-readable string.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        model_name: Name of the model being evaluated
        
    Returns:
        Formatted string with metrics
    """
    lines = [
        f"\n{'='*60}",
        f"Evaluation Results: {model_name}",
        f"{'='*60}",
        f"Total Samples:       {metrics['total_samples']}",
        f"Perfect Matches:     {metrics['perfect_matches']} "
        f"({metrics['perfect_matches']/metrics['total_samples']*100:.1f}%)",
        "",
        "OMR Normalized Edit Distance (lower is better):",
        f"  Mean:              {metrics['mean_ned']:.4f}",
        f"  Median:            {metrics['median_ned']:.4f}",
        f"  Std Dev:           {metrics['std_ned']:.4f}",
        f"  Min (Best):        {metrics['min_ned']:.4f}",
        f"  Max (Worst):       {metrics['max_ned']:.4f}",
        f"{'='*60}\n"
    ]
    return "\n".join(lines)
