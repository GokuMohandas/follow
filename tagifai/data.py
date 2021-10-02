# tagifai/data.py
# Data processing operations.

import itertools
import json
import re
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.stem import PorterStemmer
from skmultilearn.model_selection import IterativeStratification

from config import config
from tagifai import utils


def filter_items(items, include=[], exclude=[]):
    # Filter
    filtered = [item for item in items if item in include and item not in exclude]
    return filtered


def prepare(df, include=[], exclude=[], min_tag_freq=30):
    # Filter tags for each project
    df.tags = df.tags.apply(filter_items, include=include, exclude=exclude)
    tags = Counter(itertools.chain.from_iterable(df.tags.values))

    # Filter tags that have fewer than `min_tag_freq` occurrences
    tags_above_freq = Counter(tag for tag in tags.elements() if tags[tag] >= min_tag_freq)
    tags_below_freq = Counter(tag for tag in tags.elements() if tags[tag] < min_tag_freq)
    df.tags = df.tags.apply(filter_items, include=list(tags_above_freq.keys()))

    # Remove projects with no more remaining relevant tags
    df = df[df.tags.map(len) > 0]

    return df, tags_above_freq, tags_below_freq


class Stemmer(PorterStemmer):
    def stem(self, word):

        if self.mode == self.NLTK_EXTENSIONS and word in self.pool:  # pragma: no cover, nltk
            return self.pool[word]

        if self.mode != self.ORIGINAL_ALGORITHM and len(word) <= 2:  # pragma: no cover, nltk
            # With this line, strings of length 1 or 2 don't go through
            # the stemming process, although no mention is made of this
            # in the published algorithm.
            return word

        stem = self._step1a(word)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)

        return stem


def preprocess(text, lower=True, stem=False, stopwords=config.STOPWORDS):
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        stemmer = Stemmer()
        text = " ".join([stemmer.stem(word) for word in text.split(" ")])

    return text


class LabelEncoder:
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp) as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


class MultiClassLabelEncoder(LabelEncoder):
    def __str__(self):
        return f"<MultiClassLabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes


class MultiLabelLabelEncoder(LabelEncoder):
    def __str__(self):
        return f"<MultiLabelLabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(list(itertools.chain.from_iterable(y)))
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        y_one_hot = np.zeros((len(y), len(self.class_to_index)), dtype=int)
        for i, item in enumerate(y):
            for class_ in item:
                y_one_hot[i][self.class_to_index[class_]] = 1
        return y_one_hot

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            indices = np.where(np.asarray(item) == 1)[0]
            classes.append([self.index_to_class[index] for index in indices])
        return classes


def iterative_train_test_split(X, y, train_size=0.7):
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[
            1.0 - train_size,
            train_size,
        ],
    )
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


class Tokenizer:
    def __init__(self, char_level, num_tokens=None, pad_token="<PAD>", oov_token="<UNK>", token_to_index=None):
        self.char_level = char_level
        self.separator = "" if self.char_level else " "
        if num_tokens:
            num_tokens -= 2  # pad + unk tokens
        self.num_tokens = num_tokens
        self.pad_token = pad_token
        self.oov_token = oov_token
        if not token_to_index:
            token_to_index = {pad_token: 0, oov_token: 1}
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def __len__(self):
        return len(self.token_to_index)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if not self.char_level:
            texts = [text.split(" ") for text in texts]
        all_tokens = [token for text in texts for token in text]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        for token, count in counts:
            index = len(self)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            if not self.char_level:
                text = text.split(" ")
            sequence = []
            for token in text:
                sequence.append(self.token_to_index.get(token, self.token_to_index[self.oov_token]))
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for index in sequence:
                text.append(self.index_to_token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {
                "char_level": self.char_level,
                "oov_token": self.oov_token,
                "token_to_index": self.token_to_index,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp) as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def pad_sequences(sequences, max_seq_len=0):
    # Get max sequence length
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))

    # Pad
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][: len(sequence)] = sequence
    return padded_sequences


class CNNTextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, max_filter_size):
        self.X = X
        self.y = y
        self.max_filter_size = max_filter_size

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return [X, y]

    def collate_fn(self, batch):
        # Get inputs
        batch = np.array(batch, dtype=object)
        X = batch[:, 0]
        y = np.stack(batch[:, 1], axis=0)

        # Pad inputs
        X = pad_sequences(sequences=X, max_seq_len=self.max_filter_size)

        # Cast
        X = torch.LongTensor(X.astype(np.int32))
        y = torch.FloatTensor(y.astype(np.int32))

        return X, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )


def compute_features(params):
    # Set up
    utils.set_seed(seed=params.seed)

    # Load data
    projects_url = (
        "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json"
    )
    projects = utils.load_json_from_url(url=projects_url)
    df = pd.DataFrame(projects)

    # Compute features
    df["text"] = df.title + " " + df.description
    df.drop(columns=["title", "description"], inplace=True)
    df = df[["id", "created_on", "text", "tags"]]

    # Save
    features = df.to_dict(orient="records")
    df_dict_fp = Path(config.DATA_DIR, "features.json")
    utils.save_dict(d=features, filepath=df_dict_fp)

    return df, features