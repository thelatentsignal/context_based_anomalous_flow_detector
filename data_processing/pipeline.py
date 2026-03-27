""" idea: take n netflows and feed them into a bert model and learn a cls representation wtth contrastive learning.
  this should make you learn a representation for normal. """

import torch
import torch.nn as nn
from collections import defaultdict
import pandas as pd
import random
def create_augmented_ips():
    """ read in the ips, I need to build contrastive paris """
    ips = pd.read_csv('../data/processed/ips.csv').squeeze("columns")
    # 2. Group IPs by their /24 Subnet for "Positive" sampling
    subnet_map = defaultdict(list)
    for ip in ips.head(3):
        print(ip)
        parts = list(map(int, ip.split('.')))
        prefix = tuple(parts[:3])  # The first three octets
        host = parts[3]  # The last octet
        subnet_map[prefix].append(host)


    # Filter out subnets with only one IP to ensure we always have a pair
    eligible_prefixes = [k for k, v in subnet_map.items() if len(v) > 1]
    return eligible_prefixes


def get_contrastive_batch(batch_size=16):
    anchors = []
    positives = []

    # Pick random subnets and then two IPs from each
    selected_prefixes = random.choices(eligible_prefixes, k=batch_size)

    for prefix in selected_prefixes:
        # Pick two distinct hosts from the same subnet
        h1, h2 = random.sample(subnet_map[prefix], 2)

        anchors.append(list(prefix) + [h1])
        positives.append(list(prefix) + [h2])

    return torch.tensor(anchors), torch.tensor(positives)


    # Example: Get a batch for training
    # anchor_ips, positive_ips = get_contrastive_batch(batch_size=64)

class TinyIPBERT(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64):
        super().__init__()
        # Every octet (0-255) gets its own vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # A simple Transformer layer (we'll just use one for the 'start small' version)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x shape: (batch_size, 4) -> e.g., [[192, 168, 1, 1]]
        x = self.embedding(x)  # (batch_size, 4, embed_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, dim)
        out = self.transformer(x)
        cls_token = out[0]  # Take the first octet's output as the "summary"
        return cls_token

create_augmented_ips()
