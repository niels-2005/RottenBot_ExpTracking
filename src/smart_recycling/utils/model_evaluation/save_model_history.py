import mlflow
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def save_model_history(history, figsize: Tuple[int, int] = (10, 6)) -> None:
    mlflow.log_dict(history.history, "model_history.json")
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = np.arange(1, len(loss) + 1)  # Start epochs at 1

    plt.figure(figsize=figsize)

    # Plot Loss with markers
    plt.plot(
        epochs, loss, marker="o", linestyle="-", color="blue", label="Training Loss"
    )
    plt.plot(
        epochs,
        val_loss,
        marker="o",
        linestyle="-",
        color="orange",
        label="Validation Loss",
    )

    # Achsenbeschriftung
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Setze X-Ticks explizit auf ganze Zahlen
    plt.xticks(epochs)

    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/loss_curve.png")  # Log the figure to MLflow
    plt.close()
