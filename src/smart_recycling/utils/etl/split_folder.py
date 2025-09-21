import splitfolders
from typing import Tuple
from smart_recycling.utils.logging import get_configured_logger

_logger = get_configured_logger()


def split_dataset(
    source_folder: str,
    output_folder: str,
    seed: int = 42,
    ratio: Tuple[float, float, float] = (0.75, 0.1, 0.15),
) -> None:
    """Splits the dataset into training, validation, and test sets.

    Args:
        source_folder (str): Path to the source folder containing the dataset.
        output_folder (str): Path to the output folder where the split datasets will be saved.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        ratio (tuple, optional): Proportions for splitting the dataset. Defaults to (0.75, 0.1, 0.15).

    Returns:
        None

    Raises:
        Exception: Exception raised during dataset splitting.
    """
    try:
        splitfolders.ratio(source_folder, output=output_folder, seed=seed, ratio=ratio)
    except Exception as e:
        _logger.error(
            f"Error occurred while splitting dataset from {source_folder} to {output_folder}: {e}"
        )
        raise e
