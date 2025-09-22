import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mlflow
import pandas as pd


def save_wrong_predictions(
    wrong_predictions: pd.DataFrame,
    n_images: int = 20,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 12,
):
    """Plots a grid of wrong predictions and logs the figure to MLflow.

    This function takes a DataFrame of wrong predictions, displays a specified number
    of images in a grid layout, and annotates each image with the true label, predicted
    label, and prediction probability. The resulting plot is logged as an artifact
    in MLflow.

    Args:
        wrong_predictions (pd.DataFrame): A DataFrame containing wrong predictions.
            Must include columns 'file_path' (str), 'y_true_label' (str),
            'y_pred_label' (str), and 'y_pred_prob' (float).
        n_images (int, optional): The number of wrong prediction images to display.
            Defaults to 20.
        images_per_row (int, optional): The number of images per row in the grid.
            Defaults to 4.
        figsize_width (int, optional): The width of the figure in inches. Height is
            calculated automatically. Defaults to 20.
        fontsize (int, optional): Font size for the subplot titles. Defaults to 12.

    Example:
        plot_wrong_predictions(
            wrong_predictions=df,
            n_images=10,
            images_per_row=5,
            figsize_width=25,
            fontsize=10
        )
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    axes = axes.reshape(-1)

    for i, row in enumerate(wrong_predictions.iloc[:n_images].itertuples()):
        img_path = row.file_path
        true_classname = row.y_true_label
        pred_classname = row.y_pred_label
        pred_conf = row.y_pred_prob

        img = mpimg.imread(img_path)
        axes[i].imshow(img / 255.0)

        axes[i].set_title(
            f"True: {true_classname}, Pred: {pred_classname}\n prob: {pred_conf:.4f}\n",
            fontsize=fontsize,
        )

        axes[i].axis("off")

    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/wrong_predictions.png")
    plt.close()
