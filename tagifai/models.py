# tagifai/models.py
# Model architectures.

import math
from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocab_size,
        num_filters,
        filter_sizes,
        hidden_dim,
        dropout_p,
        num_classes,
        padding_idx=0,
    ):
        super().__init__()

        # Initialize embeddings
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=padding_idx,
        )

        # Conv weights
        self.filter_sizes = filter_sizes
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=f,
                )
                for f in filter_sizes
            ]
        )

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):
        # Embed
        (x_in,) = inputs
        x_in = self.embeddings(x_in)
        if not channel_first:
            x_in = x_in.transpose(1, 2)  # (N, channels, sequence length)

        z = []
        max_seq_len = x_in.shape[2]
        for i, f in enumerate(self.filter_sizes):

            # `SAME` padding
            padding_left = int(
                (
                    self.conv[i].stride[0] * (max_seq_len - 1)
                    - max_seq_len
                    + self.filter_sizes[i]
                )
                / 2
            )
            padding_right = int(
                math.ceil(
                    (
                        self.conv[i].stride[0] * (max_seq_len - 1)
                        - max_seq_len
                        + self.filter_sizes[i]
                    )
                    / 2
                )
            )

            # Conv
            _z = self.conv[i](F.pad(x_in, (padding_left, padding_right)))

            # Pool
            _z = F.max_pool1d(_z, _z.size(2)).squeeze(2)
            z.append(_z)

        # Concat outputs
        z = torch.cat(z, 1)

        # FC
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z


def initialize_model(params, vocab_size, num_classes, device=torch.device("cpu")):
    # Initialize model
    filter_sizes = list(range(1, int(params.max_filter_size) + 1))
    model = CNN(
        embedding_dim=int(params.embedding_dim),
        vocab_size=int(vocab_size),
        num_filters=int(params.num_filters),
        filter_sizes=filter_sizes,
        hidden_dim=int(params.hidden_dim),
        dropout_p=float(params.dropout_p),
        num_classes=int(num_classes),
    )
    model = model.to(device)
    return model
