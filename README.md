> You are on a previous snapshot of the main [MLOps repository](https://github.com/GokuMohandas/MLOps). This branch may only contain a subset of the larger project and is intended for viewing the iterative development process only.

## Packaging
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

## Organization
```bash
config/
├── config.py        - configuration setup
├── params.json      - training parameters
└── test_params.py   - training test parameters
tagifai/
├── data.py          - data processing components
├── eval.py          - evaluation components
├── main.py          - training/optimization pipelines
├── models.py        - model architectures
├── predict.py       - inference components
├── train.py         - training components
└── utils.py         - supplementary utilities
```

## Operations
```python linenums="1"
from pathlib import Path
from config import config
from tagifai import main

# Load data
main.load_data()

# Compute features
main.compute_features()

# Train model (test)
params_fp = Path(config.CONFIG_DIR, "test_params.json")
experiment_name = "test"
main.train_model(params_fp, experiment_name=experiment_name, run_name="model", test_run=True)

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

## Documentation
```
python -m mkdocs serve -a localhost:8000
```

## Styling
```
black .
flask8
isort .
```

## Makefile
```bash
make help
```

## CLI
```bash
tagifai --help
```

## API
```bash
# Update RUN_ID inside config/run_id.txt
uvicorn app.api:app --host 0.0.0.0 --port 5000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```
```bash
curl -X 'POST' \
  'http://localhost:5000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    {
      "text": "Transfer learning with transformers for self-supervised learning."
    },
    {
      "text": "Generative adversarial networks in both PyTorch and TensorFlow."
    }
  ]
}'
```

<!-- Citation -->
<hr>
To cite this course, please use:

```bibtex
@misc{madewithml,
    author       = {Goku Mohandas},
    title        = {Made With ML MLOps Course},
    howpublished = {\url{https://madewithml.com/}},
    year         = {2021}
}
```
