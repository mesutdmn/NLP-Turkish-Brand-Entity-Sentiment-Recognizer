import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class CustomDataset(Dataset):
    def __init__(self, texts, input_ids, attention_masks, token_type_ids, labels):
        self.texts = texts
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks
        self.labels = labels


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item ):
        text = self.texts[item]
        input_id = torch.LongTensor(self.input_ids[item])
        token_type_id = torch.LongTensor(self.token_type_ids[item])
        attention_mask = torch.LongTensor(self.attention_masks[item])
        label = torch.LongTensor(self.labels[item])


        return {
            'text': text,
            'input_ids': input_id,
            'token_type_ids': token_type_id,
            'attention_mask': attention_mask,
            'labels': label,
        }
class FeedForwardSubLayer(nn.Module):
    # Specify the two linear layers' input and output sizes
    def __init__(self, d_model, d_ff):
        super(FeedForwardSubLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

	# Apply a forward pass
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Complete the initialization of elements in the encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Set the number of attention heads
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0 #dimension, headlere tam bölünüyormu kontrol et.
        self.head_dim = d_model // num_heads
        # Set up the linear transformations
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Split the sequence embeddings in x across the attention heads
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) #.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)

    def compute_attention(self, query, key, mask=None):
        # Compute dot-product attention scores
        scores = torch.matmul(query, key.permute(0,1,3,2))
        mask = mask.unsqueeze(1).unsqueeze(1)


        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        # Normalize attention scores into attention weights
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        attention_weights = self.compute_attention(query, key, mask)

        # Multiply attention weights by values, concatenate and linearly project outputs
        output = torch.matmul(attention_weights, value)
        output = output.view(batch_size, self.num_heads, -1, self.head_dim).permute(0, 2, 1, 3).contiguous().view(
            batch_size, -1, self.d_model)
        return self.output_linear(output)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_length = max_length

        # Initialize the positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        # Calculate and assign position encodings to the matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # Update the embeddings tensor adding the positional encodings
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(100000, 512)
        self.positional_encoding = PositionalEncoder(512, 128)
        # Define a stack of multiple encoder layers
        self.layers = nn.ModuleList([EncoderLayer(512, 8, 2048, 0.1) for _ in range(6)])

    # Complete the forward pass method
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

def load_model_to_cpu(model, path="model.pth"):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    return model
