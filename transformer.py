# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Embeddings class
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len = 1000):
        super().__init__()