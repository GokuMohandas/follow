# tagifai/main.py
# Main training/optimization operations.

import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict

import mlflow
import optuna
import torch
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from tagifai import data, models, predict, train, utils

# Ignore warning
warnings.filterwarnings("ignore")


def load_data():
    # Download main data
    projects_url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json"
    projects = utils.load_json_from_url(url=projects_url)
    projects_fp = Path(config.DATA_DIR, "projects.json")
    utils.save_dict(d=projects, filepath=projects_fp)

    # Download auxiliary data
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json"
    tags = utils.load_json_from_url(url=tags_url)
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=tags, filepath=tags_fp)

    print("✅ Loaded data!")


def compute_features(params_fp=Path(config.CONFIG_DIR, "params.json")):
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Compute features
    data.compute_features(params=params)
    print("✅ Computed features!")


def optimize(
    params_fp=Path(config.CONFIG_DIR, "params.json"),
    study_name="optimization",
    num_trials=100,
):
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name, direction="maximize", pruner=pruner
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1"
    )
    study.optimize(
        lambda trial: train.objective(params, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)

    # Best trial
    print(f"Best value (f1): {study.best_trial.value}")
    params = {**params.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
    print(json.dumps(params, indent=2, cls=NumpyEncoder))


def train_model(
    params_fp=Path(config.CONFIG_DIR, "params.json"),
    experiment_name="best",
    run_name="model",
    test_run=False,
):
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        # Train
        artifacts = train.train(params=params)

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Log metrics
        performance = artifacts["performance"]
        print(json.dumps(performance["overall"], indent=2))
        metrics = {
            "precision": performance["overall"]["precision"],
            "recall": performance["overall"]["recall"],
            "f1": performance["overall"]["f1"],
            "best_val_loss": artifacts["loss"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(
                vars(artifacts["params"]), Path(dp, "params.json"), cls=NumpyEncoder
            )
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


def predict_tags(text, run_id):
    # Predict
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))

    return prediction


def params(run_id):
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = utils.load_dict(filepath=Path(artifact_uri, "params.json"))
    print(json.dumps(params, indent=2))
    return params


def performance(run_id):
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))
    print(json.dumps(performance, indent=2))
    return performance


def load_artifacts(run_id, device=torch.device("cpu")):
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


def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
    print(f"✅ Deleted experiment {experiment_name}!")