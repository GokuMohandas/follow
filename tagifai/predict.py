# tagifai/predict.py
# Prediction operations.

from distutils.util import strtobool
from typing import Dict, List

import numpy as np
import torch

from tagifai import data, train


def predict(texts, artifacts, device:torch.device = torch.device("cpu")):
    # Retrieve artifacts
    params = artifacts["params"]
    label_encoder = artifacts["label_encoder"]
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    # Prepare data
    preprocessed_texts = [
        data.preprocess(
            text,
            lower=bool(strtobool(str(params.lower))),  # params.lower could be str/bool
            stem=bool(strtobool(str(params.stem))),
        )
        for text in texts
    ]
    X = np.array(tokenizer.texts_to_sequences(preprocessed_texts), dtype="object")
    y_filler = np.zeros((len(X), len(label_encoder)))
    dataset = data.CNNTextDataset(X=X, y=y_filler, max_filter_size=int(params.max_filter_size))
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    # Get predictions
    trainer = train.Trainer(model=model, device=device)
    _, y_prob = trainer.predict_step(dataloader)
    y_pred = [np.where(prob >= float(params.threshold), 1, 0) for prob in y_prob]
    tags = label_encoder.decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "preprocessed_text": preprocessed_texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(tags))
    ]

    return predictions
