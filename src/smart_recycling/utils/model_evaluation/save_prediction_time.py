import time
import tensorflow as tf
import numpy as np
import mlflow
from smart_recycling.utils.logging import get_configured_logger

_logger = get_configured_logger()


def save_prediction_time(
    model: tf.keras.Model, test_dataset: tf.data.Dataset
) -> np.ndarray:
    """Measure and log inference time per sample in milliseconds.

    Args:
        model (tf.keras.Model): A trained TensorFlow Keras model.
        test_dataset (tf.data.Dataset): A tf.data.Dataset object for the test data.

    Raises:
        e: Raises an exception if an error occurs during prediction.

    Returns:
        np.ndarray: The predicted probabilities for each class.
    """
    try:
        try:
            num_samples = tf.data.experimental.cardinality(test_dataset).numpy()
        except:
            num_samples = sum(1 for _ in test_dataset)

        # calculate prediction time
        start_time = time.perf_counter()
        y_probs = model.predict(test_dataset, verbose=0)
        end_time = time.perf_counter()

        # calculate time per sample in milliseconds
        total_time_ms = (end_time - start_time) * 1000
        time_per_sample_ms = total_time_ms / num_samples

        mlflow.log_metric("inference_time_per_sample_ms", time_per_sample_ms)

        return y_probs

    except Exception as e:
        _logger.error(f"Error during model prediction: {e}")
        raise e
