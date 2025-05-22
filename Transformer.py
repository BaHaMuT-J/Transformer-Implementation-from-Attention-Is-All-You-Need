import numpy as np
import torch
from torch import nn

class Head(nn.Module):
  def __init__(self, embed_dim, head_dim, max_input_length, dropout=0.5, is_masked=True):
    super().__init__()
    self.query = nn.Linear(embed_dim, head_dim, bias=False)
    self.key = nn.Linear(embed_dim, head_dim, bias=False)
    self.value = nn.Linear(embed_dim, head_dim, bias=False)
    self.is_masked = is_masked
    if is_masked:
      self.register_buffer('mask', torch.tril(torch.ones(max_input_length, max_input_length)))
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, context=None):
    B, T, C = x.shape # (batch_size, time_steps, channels=embed_dim)

    if context == None:
      context = x

    Q = self.query(x)
    K = self.key(context)

    KQ = (Q @ K.transpose(-2, -1)) 
    KQ_scaled = KQ / (np.sqrt(K.shape[-1]))
    if self.is_masked:
      KQ_scaled = KQ_scaled.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
    KQ_normalized_masked_softmaxed = torch.softmax(KQ_scaled, dim=-1)
    KQ_normalized_masked_softmaxed_dropouted = self.dropout(KQ_normalized_masked_softmaxed)

    V = self.value(context)
    out = KQ_normalized_masked_softmaxed_dropouted @ V
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, embed_dim, max_input_length, dropout=0.5, is_masked=True):
    super().__init__()
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList([Head(embed_dim, head_dim, max_input_length, dropout, is_masked) for _ in range(num_heads)])
    self.linear = nn.Linear(head_dim * num_heads, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, k, context=None):
    out = torch.cat([h(k, context) for h in self.heads], dim=-1)
    out = self.linear(out)
    out = self.dropout(out)
    return out
  
class FeedFoward(nn.Module):
  def __init__(self, embed_dim, dropout=0.5):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(embed_dim, 4*embed_dim), # default: embed_dim=512, feed_forward_dim=2048)
      nn.ReLU(),
      nn.Linear(4*embed_dim, embed_dim),
      nn.Dropout(dropout),
    )

  def forward(self, x, context=None):
    return self.layers(x)
  
class SubLayer(nn.Module):
  def __init__(self, layer: nn.Module, embed_dim, dropout=0.5):
    super().__init__()
    self.Layer = layer
    self.ln = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, context=None):
    out = self.Layer(x, context)
    out = self.dropout(out)
    out = self.ln(x + out) # normalization after residual connection
    return out
  
class Encoder(nn.Module):
  def __init__(self, num_heads, embed_dim, max_input_length, dropout=0.5):
    super().__init__()
    self.multi_head_attention = SubLayer(MultiHeadAttention(num_heads, embed_dim, max_input_length, dropout, is_masked=False), embed_dim, dropout)
    self.feed_forward = SubLayer(FeedFoward(embed_dim, dropout), embed_dim, dropout)

  def forward(self, x):
    out = self.multi_head_attention(x)
    out = self.feed_forward(out)
    return out
  
class Decoder(nn.Module):
  def __init__(self, num_heads, embed_dim, max_input_length, dropout=0.5):
    super().__init__()
    self.masked_multi_head_attention = SubLayer(MultiHeadAttention(num_heads, embed_dim, max_input_length, dropout, is_masked=True), embed_dim, dropout)
    self.unmasked_multi_head_attention = SubLayer(MultiHeadAttention(num_heads, embed_dim, max_input_length, dropout, is_masked=False), embed_dim, dropout)
    self.feed_forward = SubLayer(FeedFoward(embed_dim, dropout), embed_dim, dropout)

  def forward(self, x, context=None):
    out = self.masked_multi_head_attention(x)
    
    # In Encoder-Decoder model, context is Encoder output
    # In Decoder-only model, context is None and skip this process
    if context is not None:
      out = self.unmasked_multi_head_attention(out, context)
    
    out = self.feed_forward(out)
    return out

class PositionalEncoding(nn.Module):
  def __init__(self, embed_dim, max_input_length):
    super().__init__()

    # (max_input_length, 1)
    pos = torch.arange(0, max_input_length, dtype=torch.float).unsqueeze(1)
    
    # (embed_dim/2,)
    i = torch.arange(0, embed_dim, 2).float()
    div_term = torch.exp(i * (-np.log(10000.0) / embed_dim))

    # (max_input_length, embed_dim/2)
    rate = pos * div_term

    pe = torch.zeros(max_input_length, embed_dim)
    pe[:, 0::2] = torch.sin(rate)  # even indices
    pe[:, 1::2] = torch.cos(rate)  # odd indices

    pe = pe.unsqueeze(0)  # (1, T, D) - for broadcasting over batch
    self.register_buffer('pe', pe)

  def forward(self, x):
    B, T = x.shape
    return self.pe[:, :T, :]
  
# Decoder only model
class Transformer_Decoder(nn.Module):
  def __init__(self, vocab_size, num_layers, num_heads, embed_dim, max_input_length, dropout=0.5):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_encoder = PositionalEncoding(embed_dim, max_input_length)
    self.dropout = nn.Dropout(dropout)
    self.layers = nn.Sequential(
      *[Decoder(num_heads, embed_dim, max_input_length, dropout) for _ in range(num_layers)],
      nn.Linear(embed_dim, vocab_size),
      )

  def forward(self, x):
    embed_x = self.embed(x)
    pos_embed_x = self.pos_encoder(x)
    embed_input = embed_x + pos_embed_x
    embed_input = self.dropout(embed_input)
    out = self.layers(embed_input)
    return out
  
  def generate(self, x, eos_idex, max_new_tokens=128):
    y = x

    for _ in range(max_new_tokens):
      embed_y = self.embed(y)
      pos_embed_y = self.pos_encoder(y)
      embed_target = embed_y + pos_embed_y
      embed_target = self.dropout(embed_target)

      for layer in self.layers:
        embed_target = layer(embed_target)
        
      next_token_logits = embed_target[:, -1, :]
      probs = torch.softmax(next_token_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      y = torch.cat([y, next_token], dim=1)
      if next_token.item() == eos_idex:
        break

    return y
  
# Encoder-Decoder model
class Transformer_Encoder_Decoder(nn.Module):
  def __init__(self, vocab_size, num_layers, num_heads, embed_dim, max_input_length, max_target_length, pad_idx, dropout=0.5):
    super().__init__()
    self.input_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    self.input_pos_encoder = PositionalEncoding(embed_dim, max_input_length)
    self.input_dropout = nn.Dropout(dropout)
    
    self.target_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    self.target_pos_encoder = PositionalEncoding(embed_dim, max_target_length)
    self.target_dropout = nn.Dropout(dropout)
    
    self.encoder = nn.ModuleList([Encoder(num_heads, embed_dim, max_input_length, dropout) for _ in range(num_layers)])
    self.decoder = nn.ModuleList([Decoder(num_heads, embed_dim, max_input_length, dropout) for _ in range(num_layers)])
    self.linear = nn.Linear(embed_dim, vocab_size)

  def forward(self, x, y):
    embed_x = self.input_embed(x)
    pos_embed_x = self.input_pos_encoder(x)
    embed_input = embed_x + pos_embed_x
    embed_input = self.input_dropout(embed_input)
    
    embed_y = self.target_embed(y)
    pos_embed_y = self.target_pos_encoder(y)
    embed_target = embed_y + pos_embed_y
    embed_target = self.target_dropout(embed_target)

    for layer in self.encoder:
      embed_input = layer(embed_input)
    encoder_out = embed_input

    for layer in self.decoder:
      embed_target = layer(embed_target, encoder_out)
    decoder_out = embed_target

    out = self.linear(decoder_out)
    return out
  
  def generate(self, x, sos_idx, eos_idex, max_new_tokens=128):
    # x: (1, T) - source article input

    # Encode the input
    embed_x = self.input_embed(x)
    pos_embed_x = self.input_pos_encoder(x)
    embed_input = embed_x + pos_embed_x
    embed_input = self.input_dropout(embed_input)
    for layer in self.encoder:
      embed_input = layer(embed_input)
    encoder_out = embed_input

    # Start with <sos> token
    y = torch.full((1, 1), sos_idx, dtype=torch.long, device=x.device)

    for _ in range(max_new_tokens):
      embed_y = self.target_embed(y)
      pos_embed_y = self.target_pos_encoder(y)
      embed_target = embed_y + pos_embed_y
      embed_target = self.target_dropout(embed_target)

      for layer in self.decoder:
        embed_target = layer(embed_target, encoder_out)

      logits = self.linear(embed_target)
      next_token_logits = logits[:, -1, :]  # (B, vocab_size)
      probs = torch.softmax(next_token_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
      y = torch.cat([y, next_token], dim=1)
      if next_token.item() == eos_idex:
        break

    return y