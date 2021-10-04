# tagifai/train.py
# Training operations.

import itertools
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpyencoder import NumpyEncoder
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve

from config import config
from tagifai import data, eval, models, utils


class Trainer:
    def __init__(
        self,
        model,
        device=torch.device("cpu"),
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial=None,
    ):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial

    def train_step(self, dataloader):
        # Set model to train mode
        self.model.train()
        loss = 0.0

        # Iterate over train batches
        for i, batch in enumerate(dataloader):

            # Step
            batch = [item.to(self.device) for item in batch]  # Set device
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader):
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):

                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader):
        # Set model to eval mode
        self.model.eval()
        y_trues, y_probs = [], []

        # Iterate over batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return np.vstack(y_trues), np.vstack(y_probs)

    def train(self, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
        best_model = None
        _patience = patience
        for epoch in range(num_epochs):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Pruning based on the intermediate value
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():  # pragma: no cover, optuna pruning
                    print("Unpromising trial pruned!")
                    raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:  # pragma: no cover, simple subtraction
                _patience -= 1
            if not _patience:  # pragma: no cover, simple break
                print("Stopping early!")
                break

            # Logging
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_val_loss, best_model


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true.ravel(), y_prob.ravel()
    )
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]


def train(params, trial=None):
    # Set up
    utils.set_seed(seed=params.seed)
    device = utils.set_device(cuda=params.cuda)

    # Load features
    features_fp = Path(config.DATA_DIR, "features.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    features = utils.load_dict(filepath=features_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    df = pd.DataFrame(features)
    if params.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: params.subset]  # None = all samples

    # Prepare data (filter, clean, etc.)
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=params.min_tag_freq,
    )
    params.num_samples = len(df)

    # Preprocess data
    df.text = df.text.apply(data.preprocess, lower=params.lower, stem=params.stem)

    # Encode labels
    labels = df.tags
    label_encoder = data.MultiLabelLabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)

    # Class weights
    all_tags = list(itertools.chain.from_iterable(labels.values))
    counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
    class_weights = {i: 1.0 / count for i, count in enumerate(counts)}

    # Split data
    utils.set_seed(seed=params.seed)  # needed for skmultilearn
    X = df.text.to_numpy()
    X_train, X_, y_train, y_ = data.iterative_train_test_split(
        X=X, y=y, train_size=params.train_size
    )
    X_val, X_test, y_val, y_test = data.iterative_train_test_split(X=X_, y=y_, train_size=0.5)
    test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})

    # Tokenize inputs
    tokenizer = data.Tokenizer(char_level=params.char_level)
    tokenizer.fit_on_texts(texts=X_train)
    X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
    X_val = np.array(tokenizer.texts_to_sequences(X_val), dtype=object)
    X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

    # Create dataloaders
    train_dataset = data.CNNTextDataset(
        X=X_train, y=y_train, max_filter_size=params.max_filter_size
    )
    val_dataset = data.CNNTextDataset(X=X_val, y=y_val, max_filter_size=params.max_filter_size)
    train_dataloader = train_dataset.create_dataloader(batch_size=params.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=params.batch_size)

    # Initialize model
    model = models.initialize_model(
        params=params,
        vocab_size=len(tokenizer),
        num_classes=len(label_encoder),
        device=device,
    )

    # Train model
    print(f"Parameters: {json.dumps(params.__dict__, indent=2, cls=NumpyEncoder)}")
    class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.05, patience=5
    )

    # Trainer module
    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        trial=trial,
    )

    # Train
    best_val_loss, best_model = trainer.train(
        params.num_epochs, params.patience, train_dataloader, val_dataloader
    )

    # Find best threshold
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    params.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)

    # Evaluate model
    artifacts = {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": best_model,
        "loss": best_val_loss,
    }
    device = torch.device("cpu")
    y_true, y_pred, performance = eval.evaluate(df=test_df, artifacts=artifacts)
    artifacts["performance"] = performance

    return artifacts


def objective(params, trial):
    # Paramters (to tune)
    params.embedding_dim = trial.suggest_int("embedding_dim", 128, 512)
    params.num_filters = trial.suggest_int("num_filters", 128, 512)
    params.hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train (can move some of these outside for efficiency)
    print(f"\nTrial {trial.number}:")
    print(json.dumps(trial.params, indent=2))
    artifacts = train(params=params, trial=trial)

    # Set additional attributes
    params = artifacts["params"]
    performance = artifacts["performance"]
    print(json.dumps(performance["overall"], indent=2))
    trial.set_user_attr("threshold", params.threshold)
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("f1", performance["overall"]["f1"])

    return performance["overall"]["f1"]