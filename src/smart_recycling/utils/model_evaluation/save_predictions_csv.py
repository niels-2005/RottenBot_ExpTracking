import pandas as pd
import mlflow
from smart_recycling.utils.logging import get_configured_logger

_logger = get_configured_logger()


def save_prediction_csv(
    file_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_probs: list[float],
    class_names: list[str],
) -> None:

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

        mlflow.log_text(df.to_csv(index=False), "data/complete_predictions.csv")
        mlflow.log_text(
            df_wrong_predictions.to_csv(index=False), "data/wrong_predictions.csv"
        )
    except Exception as e:
        _logger.error(f"Error occurred while saving predictions to CSV: {e}")
        raise e
