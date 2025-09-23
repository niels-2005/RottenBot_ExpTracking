import os

# needed for a bit more reproducible results when using TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mlflow
import tensorflow as tf
from rotten_bot.utils import (
    get_tensorflow_dataset,
    get_true_labels,
    compute_class_weights,
    save_confusion_matrix,
    save_model_history,
    save_prediction_csv,
    save_prediction_time,
    get_current_git_sha,
)
import numpy as np
from .experiments_config import (
    CommonConfig,
    MlflowConfig,
    DatasetConfig,
    ModelConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)
import tensorflow as tf


def run_experiment():
    # load the configurations
    common_config = CommonConfig()
    mlflow_config = MlflowConfig()
    dataset_config = DatasetConfig()
    model_config = ModelConfig()
    training_config = ModelTrainingConfig()
    evaluation_config = ModelEvaluationConfig()

    # log system metrics in the mlflow ui if enabled
    if mlflow_config.ENABLE_SYSTEM_METRICS_LOGGING:
        mlflow.enable_system_metrics_logging()

    # set the tracking uri and experiment
    mlflow.set_tracking_uri(mlflow_config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(mlflow_config.MLFLOW_EXPERIMENT_NAME)

    # Set a custom run name for better identification in the MLflow UI
    mlflow.set_tag("mlflow.runName", mlflow_config.MLFLOW_RUN_NAME)

    # set the dataset as tag in the mlflow run
    mlflow.set_tag("dataset", dataset_config.DATASET)

    # set a description for the MLflow run
    mlflow.set_tag("mlflow.note.content", mlflow_config.MLFLOW_RUN_DESCRIPTION)

    # optional log the git commit sha for better traceability
    if mlflow_config.MLFLOW_LOG_GIT_SHA:
        git_sha = get_current_git_sha()
        mlflow.set_tag("git_sha", git_sha)

    # log the experiments_config.py for future auditability
    mlflow.log_artifact(common_config.PATH_TO_CONFIG_FILE, artifact_path="config")

    # Tensorflow Dataset loading
    train_dataset = get_tensorflow_dataset(
        image_folder=f"{dataset_config.DATASET_FOLDER}/train",
        image_size=dataset_config.IMAGE_SIZE,
        batch_size=dataset_config.TRAIN_BATCH_SIZE,
        label_mode=dataset_config.LABEL_MODE,
        shuffle=True,  # shuffle True for training dataset
        seed=common_config.SEED,
    )

    val_dataset = get_tensorflow_dataset(
        image_folder=f"{dataset_config.DATASET_FOLDER}/val",
        image_size=dataset_config.IMAGE_SIZE,
        batch_size=dataset_config.VALIDATION_BATCH_SIZE,
        label_mode=dataset_config.LABEL_MODE,
        shuffle=False,  # shuffle False for validation dataset
        seed=common_config.SEED,
    )

    # setup mixed precision if enabled
    if model_config.ENABLE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = model_config.MODEL
    model.compile(
        optimizer=model_config.OPTIMIZER,
        loss=model_config.LOSS,
        metrics=model_config.METRICS,
    )

    class_weight = None
    # Optionally compute class weights to handle class imbalance
    if training_config.COMPUTE_CLASS_WEIGHTS:
        y_true = get_true_labels(train_dataset)
        class_weight = compute_class_weights(
            y_true, class_weight=training_config.CLASS_WEIGHTING_METHOD
        )

    # Actual train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=training_config.EPOCHS,
        callbacks=training_config.TRAINING_CALLBACKS,
        class_weight=class_weight,
    )

    if mlflow_config.MLFLOW_LOG_MODEL:
        # get a input example for the model signature
        for x, y in val_dataset.take(1):
            input_example = x[:1].numpy()
            break

        # # save the model history for later use
        model_history = history

        # set the history to None to avoid issues with keras model serialization.
        model.history = None
        mlflow.tensorflow.log_model(
            model,
            **mlflow_config.MLFLOW_LOG_MODEL_CONFIG,
            input_example=input_example,
        )

    # Optional evaluate the model on the test set
    if evaluation_config.INCLUDE_EVALUATION_ON_TEST_SET:
        # load the test dataset
        test_dataset = get_tensorflow_dataset(
            image_folder=f"{dataset_config.DATASET_FOLDER}/test",
            image_size=dataset_config.IMAGE_SIZE,
            batch_size=dataset_config.TEST_BATCH_SIZE,
            label_mode=dataset_config.LABEL_MODE,
            shuffle=False,  # shuffle needs to be false for later evaluation
            seed=common_config.SEED,
        )

        test_results = model.evaluate(test_dataset, return_dict=True)
        # log the test results to mlflow with a "test_" prefix
        for name, value in test_results.items():
            mlflow.log_metric(f"test_{name}", value)

        if evaluation_config.SAVE_MODEL_HISTORY:
            save_model_history(model_history)

        # optional save the prediction time to mlflow (in milliseconds)
        if evaluation_config.SAVE_PREDICTION_TIME:
            y_probs = save_prediction_time(model, test_dataset)

        # if confusion matrix or prediction csv should be saved, we need the predicted and true labels
        # additional the file paths and class names are needed
        if (
            evaluation_config.SAVE_CONFUSION_MATRIX
            or evaluation_config.SAVE_PREDICTION_CSV
        ):
            y_pred = np.argmax(y_probs, axis=1)

            y_true = get_true_labels(test_dataset)

            file_paths = test_dataset.file_paths
            class_names = test_dataset.class_names

        # optional save the confusion matrix to mlflow as plot
        if evaluation_config.SAVE_CONFUSION_MATRIX:
            save_confusion_matrix(
                y_true,
                y_pred,
                class_names,
            )

        # optional save two csv, one with all predictions and one with missclassified samples
        if evaluation_config.SAVE_PREDICTION_CSV:
            save_prediction_csv(
                file_paths,
                y_true,
                y_pred,
                y_probs,
                class_names,
                evaluation_config.SAVE_PREDICTION_CSV_THRESHOLD,
            )
