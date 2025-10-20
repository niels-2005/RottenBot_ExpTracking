import tensorflow as tf
import numpy as np
from rotten_bot.utils.logging import get_configured_logger

_logger = get_configured_logger()

def get_true_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Extract true labels from a tf.data.Dataset.

    Args:
        dataset (tf.data.Dataset): The input dataset.

    Returns:
        np.ndarray: The extracted true labels.

    Raises:
        e: Exception raised during label extraction.
    """
    try:
        y_true = np.concatenate([y for x, y in dataset], axis=0)
        return np.argmax(y_true, axis=1)
    except Exception as e:
        _logger.error(f"Error occurred while extracting true labels: {e}")
        raise e 
