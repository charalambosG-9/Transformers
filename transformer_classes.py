# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Embeddings class
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx = padding_idx)

    def forward(self, x):
        return self.embed(x)

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 1000, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False).to(x.device)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0  