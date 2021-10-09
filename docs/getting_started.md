```python linenums="1"
from pathlib import Path
from config import config
from tagifai import main, utils

# Load data
main.load_data()

# Compute features
main.compute_features()

# Train model (test)
params_fp = Path(config.CONFIG_DIR, "test_params.json")
experiment_name = "test"
main.train_model(params_fp, experiment_name=experiment_name, run_name="model")

# Delete test experiment
main.delete_experiment(experiment_name=experiment_name)
```

```bash
[01/01/20 16:36:49] INFO     ✅ Loaded data!
[01/01/20 16:36:49] INFO     ✅ Computed features!
[01/01/20 16:36:49] INFO     Run ID: b39c3a8d2c3c494984a3fa2d9d670402
[01/01/20 16:36:49] INFO     Epoch: 1 | train_loss: 0.00744, val_loss: 0.00648, lr: 1.02E-04, _patience: 10
[01/01/20 16:36:49] INFO     {
                               "precision": 0.5625,
                               "recall": 0.03125,
                               "f1": 0.05921052631578947,
                               "num_samples": 32.0
                             }
[01/01/20 16:36:49] INFO     ✅ Deleted experiment test!
```