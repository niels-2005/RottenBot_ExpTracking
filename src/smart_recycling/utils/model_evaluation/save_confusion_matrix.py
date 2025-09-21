import matplotlib.pyplot as plt
import itertools
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
from smart_recycling.utils.logging import get_configured_logger

_logger = get_configured_logger()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
) -> None:
    """Create and log a confusion matrix plot.

    Generates a confusion matrix from true and predicted labels, renders it
    using matplotlib, annotates each cell with its value (formatted as integer
    counts or as floats when normalized), logs the resulting image to the
    current MLflow run at "plots/confusion_matrix.png", and closes the figure
    to free resources.

    Args:
        y_true (np.ndarray): 1-D array of ground-truth labels.
        y_pred (np.ndarray): 1-D array of predicted labels.
        class_names (np.ndarray, optional): Sequence of class names to use for
            axis tick labels. If None, numeric indices [0..n_classes-1] are used.
            Defaults to None.
        figsize (tuple[int, int], optional): Figure size in inches as (width, height).
            Defaults to (10, 10).
        text_size (int, optional): Font size for axis ticks and cell annotation text.
            Defaults to 15.
        cmap (str, optional): Matplotlib colormap to use for the heatmap.
            Defaults to "Blues".
        norm (bool, optional): If True, normalize the confusion matrix across
            true labels (rows) using sklearn.metrics.confusion_matrix(..., normalize="true").
            When True, cell annotations are formatted as floats with two decimal places.
            Defaults to False.

    Returns:
        None: The function logs the figure to MLflow and does not return a value.
    """
    try:
        cm = (
            confusion_matrix(y_true, y_pred, normalize="true")
            if norm
            else confusion_matrix(y_true, y_pred)
        )

        # Plot the figure
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=cmap)
        fig.colorbar(cax)

        # Set class labels
        if class_names is not None:
            labels = class_names
        else:
            labels = np.arange(len(cm))

        # Set the labels and titles
        ax.set(
            title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        plt.xticks(rotation=70, fontsize=text_size)
        plt.yticks(fontsize=text_size)

        # Annotate the cells with the appropriate values
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                size=text_size,
            )

        plt.tight_layout()
        # Save the figure if requested
        mlflow.log_figure(plt.gcf(), "plots/confusion_matrix.png")
        plt.close()
    except Exception as e:
        _logger.error(f"Error occurred while making confusion matrix: {e}")
        raise e
