# tagifai/eval.py
# Evaluation components.

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from tagifai import data, train


def get_metrics(y_true, y_pred, classes, df=None):
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        metrics["class"][classes[i]] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    return metrics


def evaluate(df, artifacts, device=torch.device("cpu")):
    # Artifacts
    params = artifacts["params"]
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    label_encoder = artifacts["label_encoder"]
    model = model.to(device)
    classes = label_encoder.classes

    # Create dataloader
    X = np.array(tokenizer.texts_to_sequences(df.text.to_numpy()), dtype="object")
    y = label_encoder.encode(df.tags)
    dataset = data.CNNTextDataset(X=X, y=y, max_filter_size=int(params.max_filter_size))
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    # Determine predictions using threshold
    trainer = train.Trainer(model=model, device=device)
    y_true, y_prob = trainer.predict_step(dataloader=dataloader)
    y_pred = np.array([np.where(prob >= float(params.threshold), 1, 0) for prob in y_prob])

    # Evaluate performance
    performance = {}
    performance = get_metrics(df=df, y_true=y_true, y_pred=y_pred, classes=classes)

    return y_true, y_pred, performance
