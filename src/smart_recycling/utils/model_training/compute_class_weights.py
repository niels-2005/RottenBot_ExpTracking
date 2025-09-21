import sklearn
import numpy as np
from smart_recycling.utils.logging import get_configured_logger

_logger = get_configured_logger()


def compute_class_weights(y_true: np.ndarray, class_weight: str = "balanced") -> dict:
    """Compute class weights to handle class imbalance.

    Args:
        y_true (np.ndarray): Array of true class labels.
        class_weight (str): The type of class weight to compute. Default is "balanced".

    Returns:
        dict: A dictionary mapping class indices to their corresponding weights.

    Example usage:
        y_true = np.concatenate([y for x, y in dataset], axis=0)
        class_weights = compute_class_weight(np.argmax(y_true, axis=1), class_weight="balanced")
    """
    try:
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            class_weight=class_weight,
            classes=np.unique(y_true),
            y=y_true,
        )
        return {i: w for i, w in enumerate(class_weight)}
    except Exception as e:
        _logger.error(f"Error occurred while computing class weights: {e}")
        raise e
