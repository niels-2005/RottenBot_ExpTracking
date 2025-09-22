import mlflow
from rotten_bot.utils.logging import get_configured_logger

_logger = get_configured_logger()


class ExperimentConfig:
    # the mlflow tracking uri is the location of the mlflow server
    MLFLOW_TRACKING_URI = "http://localhost:5000"

    # the experiment where the runs will be logged
    MLFLOW_EXPERIMENT_NAME = "RottenBot-All-Classes"

    # description of the experiment, it's good to have. It'll visibly appear in the UI
    MLFLOW_EXPERIMENT_DESCRIPTION = """This experiment focuses on detecting healthy and rotten fruits and vegetables using computer vision. """

    # provide additional tags for better organization and filtering in MLflow UI
    MLFLOW_EXPERIMENT_TAGS = {
        "project_name": "rotten-bot-all-classes",
        "team": "ai-team-xyz",
        "mlflow.note.content": MLFLOW_EXPERIMENT_DESCRIPTION,
        "task": "image-classification",
        "framework": "tensorflow-keras",
        "num_classes": "28",
        "dataset": "rottenbot_all_classesv1",
        "status": "in-progress",
    }


def create_experiment():
    """Creates an MLflow experiment with specified configurations if it doesn't already exist."""
    try:
        experiment_config = ExperimentConfig()

        # connect to the MLflow tracking server
        client = mlflow.MlflowClient(tracking_uri=experiment_config.MLFLOW_TRACKING_URI)

        client.create_experiment(
            name=experiment_config.MLFLOW_EXPERIMENT_NAME,
            tags=experiment_config.MLFLOW_EXPERIMENT_TAGS,
        )
        print(
            f"Experiment '{experiment_config.MLFLOW_EXPERIMENT_NAME}' created successfully."
        )
    except Exception as e:
        _logger.info(f"Error occurred while creating experiment: {e}")
