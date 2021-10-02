# tagifai/utils.py
# Utility functions.

import json
import numbers
import random
from typing import Dict, List
from urllib.request import urlopen

import mlflow
import numpy as np
import pandas as pd
import torch


def load_json_from_url(url):
    data = json.loads(urlopen(url).read())
    return data


def load_dict(filepath):
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d, filepath, cls=None, sortkeys=False):
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def list_to_dict(list_of_dicts, key):
    d_b = {}
    for d_a in list_of_dicts:
        d_b_key = d_a.pop(key)
        d_b[d_b_key] = d_a
    return d_b


def set_seed(seed):
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def set_device(cuda):
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":  # pragma: no cover, simple tensor type setting
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device
