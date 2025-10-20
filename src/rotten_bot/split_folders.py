import splitfolders
from typing import Tuple
from rotten_bot.utils.logging import get_configured_logger

_logger = get_configured_logger()


class SplitDatasetConfig:
    # the source folder containing the original dataset, assuming it's organized in class subfolders and its not yet splitted
    SOURCE_FOLDER = "/home/ubuntu/dev/fruit_vege_disease/original_dataset/dataset"

    # the output folder where the splitted dataset will be stored
    OUTPUT_FOLDER = (
        "/home/ubuntu/dev/fruit_vege_disease/training_datasets/rottenbot_all_classesv1"
    )

    # make sure to set a seed for reproducibility
    SEED = 42

    # the ratio to split the dataset into train, val, test
    RATIO = (0.75, 0.1, 0.15)


def split_dataset(split_dataset_config: SplitDatasetConfig = SplitDatasetConfig()) -> None:
    """Splits the dataset into training, validation, and test sets.

    Args:
        split_dataset_config (SplitDatasetConfig, optional): Configuration for dataset splitting. Defaults to SplitDatasetConfig().

    Raises:
        e: Exception raised during dataset splitting.
    """
    try:
        splitfolders.ratio(
            split_dataset_config.SOURCE_FOLDER,
            output=split_dataset_config.OUTPUT_FOLDER,
            seed=split_dataset_config.SEED,
            ratio=split_dataset_config.RATIO,
        )
    except Exception as e:
        _logger.error(
            f"Error occurred while splitting dataset from {split_dataset_config.SOURCE_FOLDER} to {split_dataset_config.OUTPUT_FOLDER}: {e}"
        )
        raise e
