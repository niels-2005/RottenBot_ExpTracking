import tensorflow as tf
import numpy as np


def get_true_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Extract true labels from a tf.data.Dataset.

    Args:
        dataset (tf.data.Dataset): The input dataset.

    Returns:
        np.ndarray: The extracted true labels.
    """
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    return np.argmax(y_true, axis=1)
