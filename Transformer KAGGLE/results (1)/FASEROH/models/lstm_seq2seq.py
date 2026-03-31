
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from FASEROH.training.utils import ModelConfig

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.
    score(s_t, h_i) = v_a^T * tanh(W_s * s_t + W_h * h_i)
    """
    def __init__(self, hidden_dim: int, encoder_dim: int = None):
        super().__init__()
        encoder_dim = encoder_dim or hidden_dim
        
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(encoder_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, hidden_dim) - decoder hidden state
            keys: (batch, seq_len, encoder_dim) - encoder outputs
            mask: (batch, seq_len) - padding mask (1 for valid, 0 for pad)
        
        Returns:
            context: (batch, encoder_dim) - weighted sum of encoder outputs
            attention_weights: (batch, seq_len) - attention distribution
        """
        # Expand query to match keys sequence length
        # query: (batch, 1, hidden_dim)
        query_expanded = self.W_query(query).unsqueeze(1)
        
        # keys: (batch, seq_len, hidden_dim)
        keys_transformed = self.W_key(keys)
        
        # Calculate scores: (batch, seq_len, 1)
        scores = self.V(torch.tanh(query_expanded + keys_transformed)).squeeze(-1)
        
        # # Apply mask (if provided)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # If the mask is [batch, seq] but scores are [seq, batch], flip the mask!
        #     if mask.size(0) != scores.size(0):
        #         mask = mask.transpose(0, 1)
                
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # 1. If scores is 3D (e.g., [batch, 1, seq_len]) and mask is 2D, align dimensions
        #     if len(scores.shape) == 3 and len(mask.shape) == 2:
        #         mask = mask.unsqueeze(1)
            
        #     # 2. If the shapes are rotated/transposed (e.g., [50, 30] vs [30, 50]), flip the mask
        #     if mask.shape[-1] != scores.shape[-1]:
        #         mask = mask.transpose(-1, -2)
                
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # 1. Safely add a dimension if scores is 3D (e.g., [32, 1, 35] or [32, 35, 1])
        #     if len(scores.shape) == 3:
        #         if scores.shape[2] == 1:
        #             mask = mask.unsqueeze(2)
        #         elif scores.shape[1] == 1:
        #             mask = mask.unsqueeze(1)
            
        #     # 2. Safely transpose if batch and sequence are truly swapped
        #     if mask.shape != scores.shape:
        #         if mask.shape[0] != scores.shape[0]:
        #             mask = mask.transpose(0, 1)
                    
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # 1. Strip away all dummy dimensions (like the 1s) from both tensors
        #     sq_scores = scores.squeeze()
        #     sq_mask = mask.squeeze()
            
        #     # 2. If the core shapes are flipped, transpose the mask to match scores
        #     if sq_mask.shape != sq_scores.shape:
        #         sq_mask = sq_mask.transpose(0, 1)
                
        #     # 3. Safely mold the mask to the exact multi-dimensional shape of scores
        #     mask = sq_mask.contiguous().view(scores.shape)
            
        #     # 4. Apply the fill safely!
        #     scores = scores.masked_fill(mask == 0, -1e9)

        if mask is not None:
            # 1. PyTorch's LSTM optimizes by dropping trailing pads,
            #    shrinking the sequence length (e.g., from 50 down to 31).
            #    We must slice the mask to perfectly match this new optimized length!
            seq_len = scores.shape[-1]
            if mask.shape[-1] > seq_len:
                mask = mask[..., :seq_len]
                
            # 2. Align dimensions (if scores has a dummy 3D middle dimension)
            if len(scores.shape) == 3 and len(mask.shape) == 2:
                mask = mask.unsqueeze(1)
                
            # 3. Apply the mask safely!
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax over sequence dimension
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum: (batch, 1, seq_len) @ (batch, seq_len, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attention_weights

class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention mechanism.
    score(s_t, h_i) = s_t^T * W * h_i
    """
    def __init__(self, hidden_dim: int, encoder_dim: int = None):
        super().__init__()
        encoder_dim = encoder_dim or hidden_dim
        self.W = nn.Linear(encoder_dim, hidden_dim, bias=False)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: (batch, hidden_dim)
        # keys: (batch, seq_len, encoder_dim)
        
        # Transform keys: (batch, seq_len, hidden_dim)
        keys_transformed = self.W(keys)
        
        # Calculate scores: (batch, seq_len)
        scores = torch.bmm(keys_transformed, query.unsqueeze(-1)).squeeze(-1)
        
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # If the mask is [batch, seq] but scores are [seq, batch], flip the mask!
        #     if mask.size(0) != scores.size(0):
        #         mask = mask.transpose(0, 1)
                
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # 1. If scores is 3D (e.g., [batch, 1, seq_len]) and mask is 2D, align dimensions
        #     if len(scores.shape) == 3 and len(mask.shape) == 2:
        #         mask = mask.unsqueeze(1)
            
        #     # 2. If the shapes are rotated/transposed (e.g., [50, 30] vs [30, 50]), flip the mask
        #     if mask.shape[-1] != scores.shape[-1]:
        #         mask = mask.transpose(-1, -2)
                
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # 1. Safely add a dimension if scores is 3D (e.g., [32, 1, 35] or [32, 35, 1])
        #     if len(scores.shape) == 3:
        #         if scores.shape[2] == 1:
        #             mask = mask.unsqueeze(2)
        #         elif scores.shape[1] == 1:
        #             mask = mask.unsqueeze(1)
            
        #     # 2. Safely transpose if batch and sequence are truly swapped
        #     if mask.shape != scores.shape:
        #         if mask.shape[0] != scores.shape[0]:
        #             mask = mask.transpose(0, 1)
                    
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # if mask is not None:
        #     # 1. Strip away all dummy dimensions (like the 1s) from both tensors
        #     sq_scores = scores.squeeze()
        #     sq_mask = mask.squeeze()
            
        #     # 2. If the core shapes are flipped, transpose the mask to match scores
        #     if sq_mask.shape != sq_scores.shape:
        #         sq_mask = sq_mask.transpose(0, 1)
                
        #     # 3. Safely mold the mask to the exact multi-dimensional shape of scores
        #     mask = sq_mask.contiguous().view(scores.shape)
            
        #     # 4. Apply the fill safely!
        #     scores = scores.masked_fill(mask == 0, -1e9)

        if mask is not None:
            # 1. PyTorch's LSTM optimizes by dropping trailing pads,
            #    shrinking the sequence length (e.g., from 50 down to 31).
            #    We must slice the mask to perfectly match this new optimized length!
            seq_len = scores.shape[-1]
            if mask.shape[-1] > seq_len:
                mask = mask[..., :seq_len]
                
            # 2. Align dimensions (if scores has a dummy 3D middle dimension)
            if len(scores.shape) == 3 and len(mask.shape) == 2:
                mask = mask.unsqueeze(1)
                
            # 3. Apply the mask safely!
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attention_weights

class LSTMEncoder(nn.Module):
    """Bidirectional LSTM Encoder with optional projection"""
    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int, 
                 n_layers: int = 2, dropout: float = 0.1, 
                 bidirectional: bool = True, pad_idx: int = 0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            d_model, hidden_dim, n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # If bidirectional, project to hidden_dim for decoder init
        self.bidirectional = bidirectional
        if bidirectional:
            self.hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.cell_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """
        Args:
            src: (batch, seq_len) - token IDs
            src_mask: (batch, seq_len) - padding mask
        
        Returns:
            outputs: (batch, seq_len, hidden_dim * num_directions)
            hidden: (num_layers, batch, hidden_dim)
            cell: (num_layers, batch, hidden_dim)
        """
        embedded = self.embedding(src)  # (batch, seq_len, d_model)
        
        # Pack sequence for efficiency if mask provided
        if src_mask is not None:
            lengths = src_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # If bidirectional, concatenate directions and project for decoder init
        if self.bidirectional:
            # hidden: (num_layers * 2, batch, hidden_dim)
            batch_size = hidden.size(1)
            
            # Reshape and project
            # Separate forward and backward, then project
            hidden = hidden.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
            cell = cell.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
            
            # Concatenate forward and backward
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=-1)
            
            # Project back to hidden_dim
            hidden = self.hidden_proj(hidden)
            cell = self.cell_proj(cell)
            
            # outputs: project from hidden*2 to hidden
            outputs = self.hidden_proj(outputs)
        
        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    """LSTM Decoder with Attention mechanism"""
    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int,
                 encoder_dim: int, n_layers: int = 2, 
                 attention_type: str = "bahdanau", dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # LSTM input is embedding + context vector
        self.lstm = nn.LSTM(
            d_model + encoder_dim,  # Concatenate embedding and context
            hidden_dim, 
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        else:  # luong
            self.attention = LuongAttention(hidden_dim, encoder_dim)
        
        # Output projection: LSTM output + context -> vocab
        self.output_proj = nn.Linear(hidden_dim + encoder_dim, vocab_size)
        
        # Store attention weights for visualization
        self.attention_weights_history: List[torch.Tensor] = []
        self.store_attention = False
        
    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, 
                cell: torch.Tensor, encoder_outputs: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step forward (for autoregressive generation)
        
        Args:
            input_token: (batch, 1) - single token ID
            hidden: (n_layers, batch, hidden_dim)
            cell: (n_layers, batch, hidden_dim)
            encoder_outputs: (batch, src_len, encoder_dim)
            src_mask: (batch, src_len)
        
        Returns:
            output: (batch, vocab_size) - logits
            hidden: updated hidden state
            cell: updated cell state
        """
        # input_token: (batch, 1) -> (batch, 1, d_model)
        embedded = self.embedding(input_token)
        
        # Use last layer hidden state for attention query
        query = hidden[-1]  # (batch, hidden_dim)
        
        # Calculate attention
        context, attn_weights = self.attention(query, encoder_outputs, src_mask)
        
        # Store attention weights if visualization enabled
        if self.store_attention:
            self.attention_weights_history.append(attn_weights.detach().cpu())
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        
        # LSTM step
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Prepare for output projection: concat output and context
        output = output.squeeze(1)  # (batch, hidden_dim)
        output = torch.cat([output, context], dim=-1)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits, hidden, cell
    
    def enable_attention_storage(self, enabled: bool = True):
        """Enable/disable attention weight storage for visualization"""
        self.store_attention = enabled
        if not enabled:
            self.attention_weights_history.clear()
    
    def get_attention_matrix(self) -> Optional[np.ndarray]:
        """Get attention weights as numpy array (time_steps, src_len)"""
        if not self.attention_weights_history:
            return None
        
        # Stack along time dimension
        attn = torch.stack(self.attention_weights_history, dim=1)  # (batch, tgt_len, src_len)
        return attn.numpy()


class LSTMTaylorModel(nn.Module):
    """
    Complete LSTM Seq2Seq model with Attention for Taylor expansion.
    Includes hooks for attention visualization.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = "lstm"
        
        # Encoder
        encoder_dim = config.lstm_hidden * (2 if config.lstm_bidirectional else 1)
        self.encoder = LSTMEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            hidden_dim=config.lstm_hidden,
            n_layers=config.lstm_layers,
            dropout=config.dropout,
            bidirectional=config.lstm_bidirectional,
            pad_idx=config.pad_idx
        )
        
        # Decoder
        self.decoder = LSTMDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            hidden_dim=config.lstm_hidden,
            encoder_dim=config.lstm_hidden,  # After projection
            n_layers=config.lstm_layers,
            attention_type=config.attention_type,
            dropout=config.dropout,
            pad_idx=config.pad_idx
        )
        
        # Share embeddings if specified
        if config.share_embeddings:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Teacher forcing training forward pass.
        
        Args:
            src: (batch, src_len) - input function tokens
            tgt: (batch, tgt_len) - target Taylor expansion tokens
            src_mask: (batch, src_len) - padding mask for source
            tgt_mask: (batch, tgt_len) - padding mask for target
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) - logits
        """
        batch_size, tgt_len = tgt.size()
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_mask)
        
        # Prepare decoder inputs (shift right, start with <START>)
        # Assuming tgt is already shifted appropriately by data pipeline
        decoder_input = tgt[:, :-1]  # Exclude last token
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, tgt_len - 1, self.config.vocab_size,
                            device=src.device)
        
        # Decode step by step
        for t in range(tgt_len - 1):
            input_token = decoder_input[:, t].unsqueeze(1)
            logits, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            outputs[:, t, :] = logits
        
        return outputs
    
    def generate(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                 max_len: int = 100, start_token: int = 2, end_token: int = 3,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            src: (batch, src_len) - input function
            src_mask: (batch, src_len)
            max_len: maximum generation length
            start_token: <START> token ID
            end_token: <END> token ID
            temperature: sampling temperature (1.0 = greedy)
        
        Returns:
            generated: (batch, generated_len) - token IDs
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_mask)
        
        # Start with <START> token
        input_token = torch.full((batch_size, 1), start_token, 
                                dtype=torch.long, device=device)
        generated = [input_token]
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            logits, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            
            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            generated.append(next_token)
            
            # Check for end tokens
            finished |= (next_token.squeeze(-1) == end_token)
            if finished.all():
                break
            
            input_token = next_token
        
        return torch.cat(generated, dim=1)
    
    def get_attention_visualization_data(self, src: torch.Tensor, 
                                        tgt: torch.Tensor,
                                        src_tokens: List[str],
                                        tgt_tokens: List[str]) -> Dict:
        """
        Generate attention heatmap data for visualization.
        
        Returns dict with attention matrix and token labels.
        """
        self.decoder.enable_attention_storage(True)
        
        with torch.no_grad():
            _ = self.forward(src.unsqueeze(0), tgt.unsqueeze(0))
        
        attn_matrix = self.decoder.get_attention_matrix()[0]  # First batch item
        
        self.decoder.enable_attention_storage(False)
        
        return {
            'attention': attn_matrix,
            'source_tokens': src_tokens,
            'target_tokens': tgt_tokens[:attn_matrix.shape[0]]
        }

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
