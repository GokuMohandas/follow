# tagifai/main.py
# Main training/optimization operations.

import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

import mlflow
import optuna
import torch
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from tagifai import data, models, predict, train, utils

# Ignore warning
warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def load_data():
    """Load data from URLs and save to local drive."""
    # Download main data
    projects_url = (
        "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json"
    )
    projects = utils.load_json_from_url(url=projects_url)
    projects_fp = Path(config.DATA_DIR, "projects.json")
    utils.save_dict(d=projects, filepath=projects_fp)

    # Download auxiliary data
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json"
    tags = utils.load_json_from_url(url=tags_url)
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=tags, filepath=tags_fp)

    logger.info("✅ Loaded data!")


@app.command()
def compute_features(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
) -> None:
    """Compute and save features for training.

    Args:
        params_fp (Path, optional): Location of parameters (just using num_samples,
                                    num_epochs, etc.) to use for training.
                                    Defaults to `config/params.json`.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Compute features
    data.compute_features(params=params)
    logger.info("✅ Computed features!")


@app.command()
def optimize(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    study_name: Optional[str] = "optimization",
    num_trials: int = 100,
) -> None:
    """Optimize a subset of hyperparameters towards an objective.

    This saves the best trial's parameters into `config/params.json`.

    Args:
        params_fp (Path, optional): Location of parameters (just using num_samples,
                                    num_epochs, etc.) to use for training.
                                    Defaults to `config/params.json`.
        study_name (str, optional): Name of the study to save trial runs under. Defaults to `optimization`.
        num_trials (int, optional): Number of trials to run. Defaults to 100.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(params, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)

    # Best trial
    logger.info(f"Best value (f1): {study.best_trial.value}")
    params = {**params.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
    logger.info(json.dumps(params, indent=2, cls=NumpyEncoder))


@app.command()
def train_model(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model",
    test_run: Optional[bool] = False,
) -> None:
    """Train a model using the specified parameters.

    Args:
        params_fp (Path, optional): Parameters to use for training. Defaults to `config/params.json`.
        model_dir (Path): location of model artifacts. Defaults to config.MODEL_DIR.
        experiment_name (str, optional): Name of the experiment to save the run to. Defaults to `best`.
        run_name (str, optional): Name of the run. Defaults to `model`.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        # Train
        artifacts = train.train(params=params)

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Log metrics
        performance = artifacts["performance"]
        logger.info(json.dumps(performance["overall"], indent=2))
        metrics = {
            "precision": performance["overall"]["precision"],
            "recall": performance["overall"]["recall"],
            "f1": performance["overall"]["f1"],
            "best_val_loss": artifacts["loss"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["params"]), Path(dp, "params.json"), cls=NumpyEncoder)
            utils.save_dict(performance, Path(dp, "performance.json"))
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["params"]))

    # Save to config
    if not test_run:
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def predict_tags(text: str, run_id: str) -> Dict:
    """Predict tags for a give input text using a trained model.

    Warning:
        Make sure that you have a trained model first!

    Args:
        text (str): Input text to predict tags for.
        run_id (str): ID of the model run to load artifacts.

    Raises:
        ValueError: Run id doesn't exist in experiment.

    Returns:
        Predicted tags for input text.
    """
    # Predict
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))

    return prediction


@app.command()
def params(run_id: str) -> Dict:
    """Configured parametes for a specific run ID."""
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = utils.load_dict(filepath=Path(artifact_uri, "params.json"))
    logger.info(json.dumps(params, indent=2))
    return params


@app.command()
def performance(run_id: str) -> Dict:
    """Performance summary for a specific run ID."""
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))
    logger.info(json.dumps(performance, indent=2))
    return performance


def load_artifacts(run_id: str, device: torch.device = torch.device("cpu")) -> Dict:
    """Load artifacts for current model.

    Args:
        run_id (str): ID of the model run to load artifacts.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Artifacts needed for inference.
    """
    # Load artifacts
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    params = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "params.json")))
    label_encoder = data.MultiLabelLabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    tokenizer = data.Tokenizer.load(fp=Path(artifacts_dir, "tokenizer.json"))
    model_state = torch.load(Path(artifacts_dir, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    # Initialize model
    model = models.initialize_model(
        params=params, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    return {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "performance": performance,
    }


def delete_experiment(experiment_name: str):
    """Delete an experiment with name `experiment_name`.

    Args:
        experiment_name (str): Name of the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
    logger.info(f"✅ Deleted experiment {experiment_name}!")
