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
    def __init__(self, embed_dim, max_len = 1000, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)    
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False).to(x.device)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0  