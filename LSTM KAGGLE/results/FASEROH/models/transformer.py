
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from FASEROH.training.utils import ModelConfig

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (alternative to sinusoidal)"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        pos_emb = self.embeddings(positions)
        return self.dropout(x + pos_emb)

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer: Multi-Head Attention + FFN"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention with residual and norm
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual and norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer: Masked Self-Attn + Cross-Attn + FFN"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None):
        # Masked self-attention (causal)
        attn_out, _ = self.masked_self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention to encoder
        attn_out, attn_weights = self.cross_attn(x, encoder_out, encoder_out, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x, attn_weights

class TransformerTaylorModel(nn.Module):
    """
    Complete Transformer model for Taylor expansion.
    Supports both sinusoidal and learned positional encodings.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = "transformer"
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.decoder_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        
        # Positional encoding
        if config.positional_encoding == "sinusoidal":
            self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
            self.pos_decoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        else:  # learned
            self.pos_encoder = LearnedPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
            self.pos_decoder = LearnedPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        # Share embeddings if specified
        if config.share_embeddings:
            self.decoder_embedding.weight = self.encoder_embedding.weight
            self.output_proj.weight = self.encoder_embedding.weight  # Weight tying
        
        self._init_weights()
        self.register_buffer('causal_mask', self._generate_causal_mask(config.max_seq_len))
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # def _generate_causal_mask(self, size: int) -> torch.Tensor:
    #     """Generate causal (look-ahead) mask for decoder self-attention"""
    #     mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    #     return mask  # True where masking applies

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal (look-ahead) mask using -inf for bulletproof compatibility"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        float_mask = mask.masked_fill(mask == 1, float('-inf'))
        return float_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
            src_mask: (batch, src_len) - padding mask for source
            tgt_mask: (batch, tgt_len) - padding mask for target
        """
        batch_size, src_len = src.shape
        tgt_len = tgt.shape[1]

        # --- FIX 1: TARGET LEAKAGE (SHIFT RIGHT) ---
        # Prepend <SOS> (token 2) and drop the last token.
        # This forces the model to predict token t using only tokens up to t-1.
        start_tokens = torch.full((batch_size, 1), 2, dtype=tgt.dtype, device=tgt.device)
        tgt_input = torch.cat([start_tokens, tgt[:, :-1]], dim=1)
        
        # --- FIX 2: INVERT PADDING MASK ---
        # PyTorch expects True for PAD tokens (to ignore). Our loop gives True for VALID tokens.
        src_key_padding_mask = ~src_mask if src_mask is not None else None
        
        # Embed and add positional encoding
        src_emb = self.encoder_embedding(src) * math.sqrt(self.config.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        
        tgt_emb = self.decoder_embedding(tgt_input) * math.sqrt(self.config.d_model)
        tgt_emb = self.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create causal mask for target
        causal_mask = self.causal_mask[:tgt_len, :tgt_len].to(tgt.device)
        
        # Encode
        encoder_out = src_emb
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, src_key_padding_mask)
        
        # Decode
        decoder_out = tgt_emb
        last_attn_weights = None
        for layer in self.decoder_layers:
            decoder_out, last_attn_weights = layer(decoder_out, encoder_out, causal_mask, src_key_padding_mask)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_out)

        # --- NEW: Conditionally return the attention ---
        if return_attention:
            return logits, last_attn_weights
        
        return logits
    
    # def generate(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
    #              max_len: int = 100, start_token: int = 2, end_token: int = 3,
    #              temperature: float = 1.0) -> torch.Tensor:
    #     """
    #     Autoregressive generation using the decoder.
    #     """
    #     batch_size = src.size(0)
    #     device = src.device
        
    #     # Encode source once
    #     src_emb = self.encoder_embedding(src) * math.sqrt(self.config.d_model)
    #     src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        
    #     encoder_out = src_emb
    #     for layer in self.encoder_layers:
    #         encoder_out = layer(encoder_out, src_mask)
        
    #     # Start with <START> token
    #     generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    #     finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
    #     for i in range(max_len - 1):
    #         # Prepare target embedding with current sequence
    #         tgt_emb = self.decoder_embedding(generated) * math.sqrt(self.config.d_model)
    #         tgt_emb = self.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
    #         # Create causal mask
    #         tgt_len = generated.size(1)
    #         causal_mask = self.causal_mask[:tgt_len, :tgt_len]
            
    #         # Decode
    #         decoder_out = tgt_emb
    #         for layer in self.decoder_layers:
    #             decoder_out, _ = layer(decoder_out, encoder_out, causal_mask, src_mask)
            
    #         # Get next token logits from last position
    #         logits = self.output_proj(decoder_out[:, -1, :])
            
    #         # Sample
    #         if temperature == 0:
    #             next_token = logits.argmax(dim=-1, keepdim=True)
    #         else:
    #             probs = F.softmax(logits / temperature, dim=-1)
    #             next_token = torch.multinomial(probs, 1)
            
    #         generated = torch.cat([generated, next_token], dim=1)
            
    #         # Check for end tokens
    #         finished |= (next_token.squeeze(-1) == end_token)
    #         if finished.all():
    #             break
        
    #     return generated

    def generate(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                 max_len: int = 100, start_token: int = 2, end_token: int = 3,
                 temperature: float = 1.0) -> torch.Tensor:
        
        batch_size = src.size(0)
        device = src.device
        
        # --- FIX 2: INVERT PADDING MASK ---
        src_key_padding_mask = ~src_mask if src_mask is not None else None
        
        # Encode source once
        src_emb = self.encoder_embedding(src) * math.sqrt(self.config.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        
        encoder_out = src_emb
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, src_key_padding_mask)
            
        # Start with <START> token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # for i in range(max_len - 1):
        #     tgt_emb = self.decoder_embedding(generated) * math.sqrt(self.config.d_model)
        #     tgt_emb = self.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
        #     tgt_len = generated.size(1)
        #     causal_mask = self.causal_mask[:tgt_len, :tgt_len].to(device)
            
        #     decoder_out = tgt_emb
        #     for layer in self.decoder_layers:
        #         # Make sure to pass the inverted mask here too!
        #         decoder_out, _ = layer(decoder_out, encoder_out, causal_mask, src_key_padding_mask)
            
        #     # Get next token logits from last position
        #     logits = self.output_proj(decoder_out[:, -1, :])
            
        #     # ... [Keep the rest of your sampling logic exactly the same] ...
        #     if temperature == 0:
        #         next_token = logits.argmax(dim=-1, keepdim=True)
        #     else:
        #         probs = F.softmax(logits / temperature, dim=-1)
        #         next_token = torch.multinomial(probs, 1)
            
        #     generated = torch.cat([generated, next_token], dim=1)
            
        #     finished |= (next_token.squeeze(-1) == end_token)
        #     if finished.all():
        #         break

        for i in range(max_len - 1):
            tgt_emb = self.decoder_embedding(generated) * math.sqrt(self.config.d_model)
            tgt_emb = self.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            tgt_len = generated.size(1)
            causal_mask = self.causal_mask[:tgt_len, :tgt_len].to(device)
            
            decoder_out = tgt_emb
            for layer in self.decoder_layers:
                decoder_out, _ = layer(decoder_out, encoder_out, causal_mask, src_key_padding_mask)
            
            logits = self.output_proj(decoder_out[:, -1, :])
            
            # --- OPTIMIZATION 1: Force Greedy Search for Evaluation ---
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # --- OPTIMIZATION 2: Stop computing for finished sequences ---
            # If this sequence already generated an <EOS> token, force all future 
            # tokens to just be <PAD> (token 0) so the model stops hallucinating math!
            pad_idx = self.config.pad_idx if hasattr(self.config, 'pad_idx') else 0
            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_idx)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            finished |= (next_token.squeeze(-1) == end_token)
            if finished.all():
                break
                
        return generated

# Label Smoothing Loss (shared utility)
class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross entropy loss.
    Prevents overconfidence and improves generalization.
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing / (vocab_size - 2)  # Exclude pad and true label
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch * seq_len, vocab_size) - logits
        target: (batch * seq_len,) - token indices
        """
        # Ignore padding
        non_pad_mask = (target != self.pad_idx)
        n_tokens = non_pad_mask.sum()
        
        if n_tokens == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pred = pred[non_pad_mask]
        target = target[non_pad_mask]
        
        # Log softmax
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # KL divergence loss
        loss = F.kl_div(log_probs, true_dist, reduction='sum')
        
        return loss / n_tokens


# Visualization utilities
def plot_attention_heatmap(vis_data: Dict, save_path: Optional[str] = None):
    """
    Plot attention heatmap for LSTM model analysis.
    
    Args:
        vis_data: Dict from get_attention_visualization_data containing:
            - attention: numpy array (tgt_len, src_len)
            - source_tokens: list of strings
            - target_tokens: list of strings
    """
    attention = vis_data['attention']
    src_tokens = vis_data['source_tokens']
    tgt_tokens = vis_data['target_tokens']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, 
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'})
    plt.xlabel('Source (Input Function)')
    plt.ylabel('Target (Taylor Expansion)')
    plt.title('Attention Alignment: Function to Taylor Series')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = ModelConfig(
        vocab_size=100,
        d_model=128,
        max_seq_len=50,
        lstm_hidden=128,
        lstm_layers=2,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        share_embeddings=True,
        label_smoothing=0.1,
        positional_encoding="sinusoidal"
    )
    
    # Test LSTM model
    print("Testing LSTM Model...")
    lstm_model = TaylorModelFactory.create_model("lstm", config)
    
    # Dummy input: batch_size=2, seq_len=10
    src = torch.randint(4, 100, (2, 10))
    tgt = torch.randint(4, 100, (2, 20))
    src_mask = torch.ones(2, 10, dtype=torch.bool)
    
    output = lstm_model(src, tgt, src_mask)
    print(f"LSTM output shape: {output.shape}")  # (2, 19, vocab_size)
    
    # Test generation
    generated = lstm_model.generate(src, src_mask, max_len=20)
    print(f"Generated shape: {generated.shape}")
    
    # Test Transformer model
    print("\nTesting Transformer Model...")
    transformer_model = TaylorModelFactory.create_model("transformer", config)
    
    output = transformer_model(src, tgt, src_mask)
    print(f"Transformer output shape: {output.shape}")
    
    generated = transformer_model.generate(src, src_mask, max_len=20)
    print(f"Generated shape: {generated.shape}")
    
    # Compare parameter counts
    print("\nModel Comparison:")
    comparison = TaylorModelFactory.compare_architectures(vocab_size=1000)
    print(f"LSTM parameters: {comparison['lstm_params']:,}")
    print(f"Transformer parameters: {comparison['transformer_params']:,}")
    
    # Test label smoothing
    criterion = LabelSmoothingLoss(config.vocab_size, config.label_smoothing)
    logits = torch.randn(40, config.vocab_size)
    targets = torch.randint(0, config.vocab_size, (40,))
    loss = criterion(logits, targets)
    print(f"\nLabel smoothing loss: {loss.item():.4f}")
