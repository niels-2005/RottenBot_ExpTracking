import pandas as pd
import mlflow
from rotten_bot.utils.logging import get_configured_logger
from rotten_bot.utils import save_wrong_predictions

_logger = get_configured_logger()


def save_prediction_csv(
    file_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_probs: list[float],
    class_names: list[str],
    threshold: float,
) -> None:
    """Saves the model predictions to a CSV file.

    The function creates a DataFrame containing the file paths, true labels,
    predicted labels, predicted probabilities, and whether the prediction was correct.
    It then saves two CSV files: one with all predictions and another with only the
    misclassified samples. Both files are logged as artifacts in MLflow.

    Args:
        file_paths (list[str]): List of file paths for the images.
        y_true (list[int]): List of true class indices.
        y_pred (list[int]): List of predicted class indices.
        y_probs (list[float]): List of predicted probabilities for each class.
        class_names (list[str]): List of class names corresponding to the class indices.

    Raises:
        e: If an error occurs while saving the predictions.
    """
    try:
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_prob": y_probs.max(axis=1).round(4),
                "y_true_label": [class_names[i] for i in y_true],
                "y_pred_label": [class_names[i] for i in y_pred],
            }
        )
        df["pred_correct"] = df["y_true"] == df["y_pred"]

        df_wrong_predictions = df[df["pred_correct"] == False].sort_values(
            by="y_pred_prob", ascending=False
        )

        df_wrong_predictions_threshold = df_wrong_predictions[
            df_wrong_predictions["y_pred_prob"] >= threshold
        ]

        save_wrong_predictions(
            wrong_predictions=df_wrong_predictions_threshold,
            n_images=len(df_wrong_predictions_threshold),
        )

        mlflow.log_text(df.to_csv(index=False), "data/complete_predictions.csv")
        mlflow.log_text(
            df_wrong_predictions.to_csv(index=False), "data/wrong_predictions.csv"
        )
        mlflow.log_text(
            df_wrong_predictions_threshold.to_csv(index=False),
            "data/wrong_predictions_threshold.csv",
        )
    except Exception as e:
        _logger.error(f"Error occurred while saving predictions to CSV: {e}")
        raise e
