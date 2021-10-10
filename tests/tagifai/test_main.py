# tests/app/test_cli.py
# Test app/cli.py components.

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from config import config
from tagifai import main
from tagifai.main import app

runner = CliRunner()


def test_load_data():
    result = runner.invoke(app, ["load-data"])
    assert result.exit_code == 0


def test_compute_features():
    result = runner.invoke(app, ["compute-features"])
    assert result.exit_code == 0


@pytest.mark.training
def test_optimize():
    study_name = "test_optimization"
    result = runner.invoke(
        app,
        [
            "optimize",
            "--params-fp",
            f"{Path(config.CONFIG_DIR, 'test_params.json')}",
            "--study-name",
            f"{study_name}",
            "--num-trials",
            1,
        ],
    )
    assert result.exit_code == 0

    # Delete study
    main.delete_experiment(experiment_name=study_name)
    shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))


@pytest.mark.training
def test_train_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    result = runner.invoke(
        app,
        [
            "train-model",
            "--params-fp",
            f"{Path(config.CONFIG_DIR, 'test_params.json')}",
            "--experiment-name",
            f"{experiment_name}",
            "--run-name",
            f"{run_name}",
            "--test-run"
        ],
    )
    assert result.exit_code == 0

    # Delete experiment
    main.delete_experiment(experiment_name=experiment_name)
    shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))


def test_predict_tags():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    result = runner.invoke(app, ["predict-tags", "Transfer learning with BERT.", f"{run_id}"])
    assert result.exit_code == 0


def test_params():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    result = runner.invoke(app, ["params", f"{run_id}"])
    assert result.exit_code == 0


def test_performance():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    result = runner.invoke(app, ["performance", f"{run_id}"])
    assert result.exit_code == 0
