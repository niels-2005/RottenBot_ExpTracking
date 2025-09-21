import tensorflow as tf
import mlflow

# This Config file is used to configure the experiments
# it'll be imported in the run_experiments.ipynb notebook and affect the whole experiment run
# the file will be logged in MLflow for future auditability, so make sure to use only relevant configurations.


class CommonConfig:
    # the seed is used for reproducibility (note that in tensorflow exact reproducibility is not guaranteed)
    # we'll get as close as possible
    SEED = 42

    # Path to this config file, used for logging the config file in MLflow (for future auditability)
    PATH_TO_CONFIG_FILE = "experiments_config.py"


class MlflowConfig:
    # the mlflow tracking uri is the location of the mlflow server
    MLFLOW_TRACKING_URI = "http://localhost:5000"

    # optional enable system metrics logging (CPU, RAM, GPU usage), you can inspect these metrics in the MLflow UI
    ENABLE_SYSTEM_METRICS_LOGGING = False

    # the experiment where the runs will be logged, make sure to create the experiment with the utils function
    # in experiment_utils.ipynb
    MLFLOW_EXPERIMENT_NAME = "Smart_Recycling_AI"

    # set the mlflow run name, make sure it's unique for each run (IMPORTANT for better tracking in the MLflow UI)
    MLFLOW_RUN_NAME = "EfficientNetV2B0_Freezed"

    # you can set a run description, it's good to have. It'll visibly appear in the UI
    MLFLOW_RUN_DESCRIPTION = "EfficientNetV2B0 with freezed backbone, data augmentation and class weights. Adam and CategoricalCrossentropy."

    # optional log the git commit sha, it's really useful for traceability
    # and recommend, because codebase can change over time
    MLFLOW_LOG_GIT_SHA = True

    # optional log the model to mlflow, this is useful for model deployment later on
    # disable it when you perform initial experiments to save time and space
    MLFLOW_LOG_MODEL = True

    # the configuration for the logged model, it's only used if MLFLOW_LOG_MODEL is True
    # note that the model and signature are dynamically set during the run
    # the "name" is the name of the model in mlflow, you can change it to whatever you want
    # the "registered_model_name" is the name of the registered model in mlflow, it can be None if you don't want to register the model
    # the "keras_model_kwargs" are the arguments passed to the keras model save function
    MLFLOW_LOG_MODEL_CONFIG = {
        "name": "model",
        "keras_model_kwargs": {"save_format": "keras"},
        "registered_model_name": None,
    }


class DatasetConfig:
    # the dataset will be tracked as a "tag" in the mlflow run
    # if you change the dataset, change also the tag accordingly
    DATASET = "garbage-dataset-v1"

    # the path to the dataset folder, make sure its already been splitted in train, val, test folders
    # if not, you can use the utils function in the experiment_utils.ipynb notebook to split it.
    DATASET_FOLDER = f"/home/ubuntu/dev/smart_recycling/{DATASET}"

    # the image size to resize the images for model training
    IMAGE_SIZE = (224, 224)

    # the batch size for training, validation and test datasets
    # the train batch size can affect the model performance, increase for faster training.
    # The validation and test batch sizes can be kept low to save memory or high to speed up evaluation
    TRAIN_BATCH_SIZE = 64
    VALIDATION_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32

    # the label mode, examples are "categorical", "binary". depends on the task
    LABEL_MODE = "categorical"


class ModelConfig:
    # optional enable mixed precision. Reference: https://www.tensorflow.org/guide/mixed_precision
    # if its set to True, make sure your last layer in the model has dtype="float32". THATS IMPORTANT!
    ENABLE_MIXED_PRECISION = True

    BACKBONE = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None,
        include_preprocessing=True,
    )
    BACKBONE.trainable = False

    # the model architecture, you can change it to experiment with different architectures
    # and evaluate their performance in the MLflow UI
    # make sure to set seeds where possible for near reproducibility
    MODEL = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3)),
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
            tf.keras.layers.RandomRotation(0.12, seed=42),
            tf.keras.layers.RandomZoom(0.12, seed=42),
            tf.keras.layers.RandomContrast(0.12, seed=42),
            tf.keras.layers.RandomBrightness(0.12, seed=42),
            BACKBONE,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(8, activation="softmax", dtype="float32"),
        ]
    )

    # the model compilation parameters
    # you can change the optimizer, loss function and metrics to experiment
    # you can have as much metrics as you want. Just make sure that they are placed in a list.
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    METRICS = [tf.keras.metrics.F1Score(average="weighted")]


class ModelTrainingConfig:
    # the number of epochs to train the model, if you use early stopping as callback, it can be kept high
    EPOCHS = 50

    # optionally compute the class weights, its recommend when you're dealing with imbalanced datasets
    # its recommend to use "balanced" method, but you can experiment with different methods
    COMPUTE_CLASS_WEIGHTS = True
    CLASS_WEIGHTING_METHOD = "balanced"

    EARLY_STOPPING_CALLBACK = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    REDUCE_LR_ON_PLATEAU_CALLBACK = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7
    )

    MLFLOW_CALLBACK = mlflow.tensorflow.MlflowCallback()

    # define the training callbacks, you can add as much as you want.
    # the reference for some callbacks in tensorflow: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
    TRAINING_CALLBACKS = [
        EARLY_STOPPING_CALLBACK,
        REDUCE_LR_ON_PLATEAU_CALLBACK,
        MLFLOW_CALLBACK,
    ]


class ModelEvaluationConfig:
    # optionally evaluate the model on the test set
    INCLUDE_EVALUATION_ON_TEST_SET = True

    # optionally save the model history as a plot and a csv file
    # the model history is the training and validation metrics and loss values for each epoch
    # It will only computed if INCLUDE_EVALUATION_ON_TEST_SET is True
    SAVE_MODEL_HISTORY = True

    # optinally save the average prediction time per sample on the test set
    # its been computed in milliseconds (ms), if you have strict requirements on prediction time
    # It will only computed if INCLUDE_EVALUATION_ON_TEST_SET is True
    SAVE_PREDICTION_TIME = True

    # optionally save the confusion matrix as a plot, it's really recommended to enable it
    # to better understand the model performance on each class
    # It will only computed if INCLUDE_EVALUATION_ON_TEST_SET is True
    SAVE_CONFUSION_MATRIX = True

    # optionally save all predictions with true labels, predicted labels, predicted probabilities, file_paths
    # it saves two csv files, one with alle the predictions and one with only the misclassified samples
    # It will only computed if INCLUDE_EVALUATION_ON_TEST_SET is True
    SAVE_PREDICTION_CSV = True
