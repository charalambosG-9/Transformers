# Imports
import torch
import torch.nn as nn
import math
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Embeddings class
class Embeddings(nn.Module):
    '''
    Args:
        vocab_size (int): Size of the dictionary of embeddings
        d_model (int): The size of each embedding vector
        padding_idx (int): If specified, represents the index of the padding, which is not updated durirng training.

    Function creates a simple lookup table that stores embeddings of a fixed dictionary and size. 
    '''
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoding(nn.Module):
    '''
    Args:
        d_model (int): The size of each embedding vector.
        max_len (int, optional): The maximum length of the input sequences. Default is 1000.
        dropout (float, optional): Dropout probability to apply after adding positional encodings. Default is 0.1.

    Function adds positional information to token embeddings to enable the model
    to take into account the order of tokens in the sequence. It uses fixed sinusoidal
    functions (sine for even dimensions, cosine for odd dimensions) as described in
    the original Transformer paper "Attention Is All You Need".

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    '''
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize positional encodings to all zeros
        pe = torch.zeros(max_len, d_model)

        # Create a tensor of shape (max_len, 1) with values [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute the div_term used to scale the position indices in the sine/cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the embedding dimension
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the embedding dimension
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension so pe has shape (1, max_len, d_model)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):

        # Add positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False).to(x.device)

        # Apply dropout and return the result
        return self.dropout(x)


# Position-wise Feed Forward class
class PositionWiseFeedForward(nn.Module):
    '''

    Args:
        d_model (int): The size of each embedding vector.
        d_ff (int): The hidden layer size in the feed-forward network.

    FeedForward Neural network

    Shape:
        - Input: (batch_size, d_model, d_ff)
        - Output: (batch_size, d_model, d_ff)
    '''
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Multi-headed Attention class
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.cross_attn = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = Embeddings(src_vocab_size, d_model)
        self.decoder_embedding = Embeddings(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask.to(src.device), tgt_mask.to(tgt.device)

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

# Hyperparameters
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Model, criterion, optimizer
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Dummy data on device
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)).to(device)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device)

# Training loop
transformer.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
