import numpy as np
from typing import List

def compute_ece(confidences: List[float], correct_labels: List[bool], n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    
    Args:
        confidences: list of max softmax probabilities (0 to 1)
        correct_labels: list of booleans (was prediction correct?)
        n_bins: number of bins to use for ECE calculation
        
    Returns:
        float: Expected Calibration Error (lower is better, 0 = perfectly calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        
        # In the top bin, include the upper boundary
        if i == n_bins - 1:
            mask = (np.array(confidences) >= lo) & (np.array(confidences) <= hi)
        else:
            mask = (np.array(confidences) >= lo) & (np.array(confidences) < hi)
            
        bin_count = mask.sum()
        if bin_count == 0:
            continue
            
        bin_conf = np.array(confidences)[mask].mean()
        bin_acc = np.array(correct_labels)[mask].mean()
        
        ece += bin_count * abs(bin_acc - bin_conf)
        
    return ece / len(confidences)
