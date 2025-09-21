import tensorflow as tf
from typing import Tuple
from smart_recycling.utils.logging import get_configured_logger

_logger = get_configured_logger()


def get_tensorflow_dataset(
    image_folder: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 64,
    label_mode: str = "categorical",
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from a directory of images.

    Args:
        image_folder (str): Path to the folder containing images.
        image_size (tuple, optional): Size to resize images. Defaults to (224, 224).
        batch_size (int, optional): Number of images per batch. Defaults to 64.
        label_mode (str, optional): Type of label encoding. Defaults to "categorical".
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        seed (int, optional): Random seed for shuffling. Defaults to 42.

    Returns:
        tf.data.Dataset: A TensorFlow dataset.
    """
    try:
        return tf.keras.utils.image_dataset_from_directory(
            image_folder,
            image_size=image_size,
            batch_size=batch_size,
            label_mode=label_mode,
            shuffle=shuffle,
            seed=seed,
        )
    except Exception as e:
        _logger.error(
            f"Error occurred while creating TensorFlow dataset from {image_folder}: {e}"
        )
        raise e
