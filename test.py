from pathlib import Path
from config import config
from tagifai import main, utils

# Load auxiliary data
main.download_auxiliary_data()

# Compute features
main.compute_features()

# Train model (test)
params_fp = Path(config.CONFIG_DIR, "test_params.json")
experiment_name = "test"
main.train_model(params_fp, experiment_name=experiment_name, run_name="model")

# Delete test experiment
main.delete_experiment(experiment_name=experiment_name)