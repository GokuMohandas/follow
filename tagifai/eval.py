# tagifai/eval.py
# Evaluation components.

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function

from tagifai import data, train


@slicing_function()
def cv_transformers(x):
    """Projects with the `computer-vision` and `transformers` tags."""
    return all(tag in x.tags for tag in ["computer-vision", "transformers"])


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 7  # less than 7 words


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List, df: pd.DataFrame = None):
    """Calculate metrics for fine-grained performance evaluation.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        classes (List): List of all unique classes.
        df (pd.DataFrame, optional): dataframe used for slicing.

    Returns:
        Dictionary of fine-grained performance metrics.
    """
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

    # Slicing metrics
    if df is not None:
        # Slices
        slicing_functions = [cv_transformers, short_text]
        applier = PandasSFApplier(slicing_functions)
        slices = applier.apply(df)

        # Score slices
        # Use snorkel.analysis.Scorer for multiclass tasks
        # Naive implementation for our multilabel task
        # based on snorkel.analysis.Scorer
        metrics["slices"] = {}
        metrics["slices"]["class"] = {}
        for slice_name in slices.dtype.names:
            mask = slices[slice_name].astype(bool)
            if sum(mask):  # pragma: no cover, test set may not have enough samples for slicing
                slice_metrics = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average="micro"
                )
                metrics["slices"]["class"][slice_name] = {}
                metrics["slices"]["class"][slice_name]["precision"] = slice_metrics[0]
                metrics["slices"]["class"][slice_name]["recall"] = slice_metrics[1]
                metrics["slices"]["class"][slice_name]["f1"] = slice_metrics[2]
                metrics["slices"]["class"][slice_name]["num_samples"] = len(y_true[mask])

        # Weighted overall slice metrics
        metrics["slices"]["overall"] = {}
        for metric in ["precision", "recall", "f1"]:
            metrics["slices"]["overall"][metric] = np.mean(
                list(
                    itertools.chain.from_iterable(
                        [
                            [metrics["slices"]["class"][slice_name][metric]]
                            * metrics["slices"]["class"][slice_name]["num_samples"]
                            for slice_name in metrics["slices"]["class"]
                        ]
                    )
                )
            )

    return metrics


def evaluate(df: pd.DataFrame, artifacts: Dict, device: torch.device = torch.device("cpu")):
    """Evaluate performance on data.

    Args:
        df (pd.DataFrame): Dataframe (used for slicing).
        artifacts (Dict): Artifacts needed for inference.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Ground truth and predicted labels, performance.
    """
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
