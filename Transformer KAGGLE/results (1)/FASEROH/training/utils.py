
from dataclasses import dataclass
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ModelConfig:
    """Unified configuration for both LSTM and Transformer models"""
    vocab_size: int = 1000
    d_model: int = 256          # Embedding dimension
    max_seq_len: int = 100      # Maximum sequence length
    
    # LSTM specific
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_bidirectional: bool = True
    attention_type: str = "bahdanau"  # "bahdanau" or "luong"
    
    # Transformer specific
    n_layers: int = 4           # Encoder/decoder layers
    n_heads: int = 8            # Attention heads
    d_ff: int = 1024            # Feed-forward dimension
    dropout: float = 0.1
    positional_encoding: str = "sinusoidal"  # "sinusoidal" or "learned"
    
    # Shared settings
    share_embeddings: bool = True
    label_smoothing: float = 0.1
    pad_idx: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'lstm_bidirectional': self.lstm_bidirectional,
            'attention_type': self.attention_type,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'positional_encoding': self.positional_encoding,
            'share_embeddings': self.share_embeddings,
            'label_smoothing': self.label_smoothing,
            'pad_idx': self.pad_idx
        }

class TaylorModelFactory:
    """Factory to create models with identical interface"""
    
    @staticmethod
    # def create_model(model_type: str, config: ModelConfig) -> Union[LSTMTaylorModel, TransformerTaylorModel]:
    def create_model(model_type: str, config: ModelConfig):

        """
        Create model instance based on type.
        
        Args:
            model_type: "lstm" or "transformer"
            config: Model configuration
        
        Returns:
            Model instance with forward/generge interface
        """
        if model_type.lower() == "lstm":
            from FASEROH.models.lstm_seq2seq import LSTMTaylorModel
            return LSTMTaylorModel(config)
        elif model_type.lower() == "transformer":
            from FASEROH.models.transformer import TransformerTaylorModel
            return TransformerTaylorModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def compare_architectures(vocab_size: int = 1000) -> Dict[str, int]:
        """
        Print parameter counts for both architectures.
        Useful for model comparison in reports.
        """
        config = ModelConfig(vocab_size=vocab_size)
        
        lstm_model = TaylorModelFactory.create_model("lstm", config)
        transformer_model = TaylorModelFactory.create_model("transformer", config)
        
        lstm_params = sum(p.numel() for p in lstm_model.parameters())
        trans_params = sum(p.numel() for p in transformer_model.parameters())
        
        return {
            'lstm_params': lstm_params,
            'transformer_params': trans_params,
            'ratio': trans_params / lstm_params if lstm_params > 0 else 0
        }

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict
import copy
import warnings

import sys
import os
sys.path.append("/teamspace/studios/this_studio/")

# For BLEU score calculation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. BLEU scores will not be calculated.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 100
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # Learning rate scheduling
    lr_scheduler: str = "onecycle"  # "reduce_on_plateau", "cosine_warm_restarts", "none"
    lr_patience: int = 5  # for ReduceLROnPlateau
    lr_factor: float = 0.5  # for ReduceLROnPlateau
    T_0: int = 10  # for CosineAnnealingWarmRestarts (first restart)
    T_mult: int = 2  # for CosineAnnealingWarmRestarts (multiplication factor)
    
    # Early stopping
    early_stopping_patience: int = 10
    monitor_metric: str = "loss"  # Checkpoint based on this
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_epochs: List[int] = field(default_factory=lambda: [10, 20, 30, 40])
    # Each entry is epoch to switch to next complexity level
    # Level 0: Polynomials only
    # Level 1: Add Basic functions
    # Level 2: Add Binary operations
    # Level 3: Add Compositions
    # Level 4: Add Complex functions
    
    # Checkpointing
    checkpoint_dir: str = "/kaggle/working/checkpoints/"
    save_best_only: bool = True
    
    # Validation
    validation_split: float = 0.2
    validate_every: int = 1  # Validate every N epochs
    
    # Teacher forcing
    teacher_forcing_ratio: float = 1.0  # Always 1.0 for training (handled by model)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'gradient_clip_norm': self.gradient_clip_norm,
            'lr_scheduler': self.lr_scheduler,
            'early_stopping_patience': self.early_stopping_patience,
            'monitor_metric': self.monitor_metric,
            'use_curriculum': self.use_curriculum,
        }


class MetricsCalculator:
    """Calculate and track all training metrics"""
    
    def __init__(self, pad_idx: int = 0, vocab=None):
        self.pad_idx = pad_idx
        self.vocab = vocab  # For BLEU score detokenization if needed
        self.reset()
        
        if NLTK_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
    
    def reset(self):
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_sequences = 0
        self.correct_sequences = 0
        self.total_loss = 0.0
        self.batch_count = 0
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: float, mask: Optional[torch.Tensor] = None):
        """
        Update metrics with batch results.
        
        Args:
            predictions: (batch, seq_len) predicted token IDs
            targets: (batch, seq_len) target token IDs
            loss: batch loss value
            mask: (batch, seq_len) valid token mask
        """
        self.total_loss += loss
        self.batch_count += 1
        
        if mask is None:
            mask = (targets != self.pad_idx)
        
        # Token accuracy (excluding padding)
        valid_tokens = mask.sum().item()
        correct = ((predictions == targets) & mask).sum().item()
        self.correct_tokens += correct
        self.total_tokens += valid_tokens
        
        # Sequence accuracy (exact match, excluding padding)
        # Check if all valid tokens match
        matches = (predictions == targets) | ~mask
        seq_correct = matches.all(dim=1).sum().item()
        self.correct_sequences += seq_correct
        self.total_sequences += predictions.size(0)
        
        # Store for BLEU calculation (do it at end for efficiency)
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        metrics = {}
        
        # Loss
        metrics['loss'] = self.total_loss / max(self.batch_count, 1)
        
        # Token accuracy
        metrics['token_accuracy'] = self.correct_tokens / max(self.total_tokens, 1)
        
        # Sequence accuracy
        metrics['sequence_accuracy'] = self.correct_sequences / max(self.total_sequences, 1)
        
        # Perplexity
        try:
            metrics['perplexity'] = math.exp(metrics['loss'])
        except OverflowError:
            metrics['perplexity'] = float('inf')
        
        # BLEU score (if available)
        if NLTK_AVAILABLE and self.vocab is not None:
            bleu_scores = []
            for pred, tgt in zip(self.all_predictions, self.all_targets):
                # Remove padding and convert to lists
                pred_tokens = [int(p) for p in pred if p != self.pad_idx]
                tgt_tokens = [int(t) for t in tgt if t != self.pad_idx]
                
                if len(pred_tokens) > 0 and len(tgt_tokens) > 0:
                    score = sentence_bleu(
                        [tgt_tokens], 
                        pred_tokens,
                        smoothing_function=self.smoothing
                    )
                    bleu_scores.append(score)
            
            metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
        else:
            metrics['bleu'] = 0.0
        
        return metrics


class CurriculumScheduler:
    """
    Curriculum learning scheduler.
    Progressively increases difficulty based on epochs.
    """
    
    def __init__(self, schedule: Optional[List[int]] = None):
        """
        Args:
            schedule: List of epoch numbers where difficulty increases.
                     At epoch schedule[0], allow level 1, etc.
        """
        self.schedule = schedule or [10, 20, 30, 40]
        self.current_level = 0
        self.complexity_levels = ['polynomial', 'basic', 'binary', 'composition', 'complex']
        
    def get_allowed_complexities(self, epoch: int) -> List[str]:
        """Get list of allowed complexity classes for current epoch"""
        level = 0
        for threshold in self.schedule:
            if epoch >= threshold:
                level += 1
        
        level = min(level, len(self.complexity_levels) - 1)
        self.current_level = level
        
        # Return all complexities up to current level
        return self.complexity_levels[:level + 1]
    
    def filter_dataset(self, dataset, epoch: int):
        """
        Filter dataset to include only samples up to current complexity level.
        Assumes dataset items have 'complexity_class' field.
        """
        allowed = self.get_allowed_complexities(epoch)
        indices = []
        
        for i, item in enumerate(dataset):
            # Handle both dict-style and object-style datasets
            if isinstance(item, dict):
                complexity = item.get('complexity_class', 'basic')
            else:
                complexity = getattr(item, 'complexity_class', 'basic')
            
            if complexity in allowed:
                indices.append(i)
        
        return indices
    
    def __repr__(self):
        return f"CurriculumScheduler(current_level={self.current_level}, allowed={self.get_allowed_complexities(0)})"


class EarlyStopping:
    """Early stopping based on monitored metric (higher is better for accuracy)"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'max', verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, metric_value: float) -> bool:
        if self.best_value is None:
            self.best_value = metric_value
            return False
        
        if self.mode == 'max':
            improved = metric_value > self.best_value + self.min_delta
        else:
            improved = metric_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
            if self.verbose:
                logger.info(f"Metric improved to {metric_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"No improvement for {self.counter} epochs. Best: {self.best_value:.4f}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered!")
        
        return self.early_stop


class CheckpointManager:
    """Manage model checkpointing based on monitored metric"""
    
    def __init__(self, checkpoint_dir: str, monitor: str = 'val_seq_accuracy', 
                 mode: str = 'max', save_best_only: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_path = None
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_value': self.best_value
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best if improved
        if is_best:
            self.best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, self.best_path)
            logger.info(f"Saved best checkpoint with {self.monitor}={metrics[self.monitor]:.4f}")
        
        # Save periodic (every 10 epochs)
        if epoch % 10 == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)
    
    def check_improvement(self, metric_value: float) -> bool:
        """Check if metric improved"""
        if self.mode == 'max':
            improved = metric_value > self.best_value
        else:
            improved = metric_value < self.best_value
        
        if improved:
            self.best_value = metric_value
        
        return improved
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, 
                       checkpoint_path: Optional[str] = None) -> Dict:
        """Load checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.best_path or (self.checkpoint_dir / 'checkpoint_latest.pt')
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


class Trainer:
    """
    Complete training pipeline with curriculum learning, LR scheduling,
    gradient clipping, and comprehensive metrics.
    """
    
    def __init__(self, model, config: TrainingConfig, device: str = 'auto'):
        self.model = model
        self.config = config
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # LR Scheduler
        self.scheduler = self._init_scheduler()
        
        # Loss function with label smoothing
        self.criterion = self._init_criterion()
        
        # Curriculum learning
        self.curriculum = CurriculumScheduler(config.curriculum_epochs) if config.use_curriculum else None
        
        # Early stopping (monitor sequence accuracy, higher is better)
        mode = 'max' if 'accuracy' in config.monitor_metric else 'min'
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=mode
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            monitor=config.monitor_metric,
            mode=mode,
            save_best_only=config.save_best_only
        )
        
        # Metrics tracking
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.current_epoch = 0
        
    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config.lr_scheduler == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max' if 'accuracy' in self.config.monitor_metric else 'min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience
                # verbose=True
            )
        elif self.config.lr_scheduler == 'cosine_warm_restarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult
            )
        else:
            return None
    
    def _init_criterion(self):
        """Initialize loss function"""
        # Use label smoothing if specified
        if self.config.label_smoothing > 0:
            from FASEROH.training.utils import LabelSmoothingLoss  # Assuming from previous phase
            return LabelSmoothingLoss(
                vocab_size=self.model.config.vocab_size,
                smoothing=self.config.label_smoothing,
                pad_idx=self.model.config.pad_idx
            )
        else:
            return nn.CrossEntropyLoss(ignore_index=self.model.config.pad_idx)

    def _apply_curriculum(self, train_dataset, epoch: int):
        """Apply curriculum learning by filtering dataset"""
        if self.curriculum is None:
            return train_dataset
        
        allowed = self.curriculum.get_allowed_complexities(epoch)
        logger.info(f"Epoch {epoch}: Curriculum level allows {allowed}")
        
        # Filter indices
        indices = self.curriculum.filter_dataset(train_dataset, epoch)
        
        if len(indices) == 0:
            logger.warning("No samples match curriculum criteria, using full dataset")
            return train_dataset
        
        logger.info(f"Curriculum filtered dataset: {len(indices)}/{len(train_dataset)} samples")
        return torch.utils.data.Subset(train_dataset, indices)
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics_calc = MetricsCalculator(pad_idx=self.model.config.pad_idx)
        
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # batch = {k: v.to(self/.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Move to device
            src = batch['encoder_input'].to(self.device)
            tgt = batch['decoder_target'].to(self.device)
            # src_mask = batch.get('encoder_mask', None)
            src_mask = (src != self.model.config.pad_idx).to(self.device)
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            
            # # Forward pass
            # self.optimizer.zero_grad()
            
            # # Model forward returns logits
            # logits = self.model(src, tgt, src_mask)

            # seq_len = logits.size(1)
            # targets_aligned = tgt[:, -seq_len:]
            
            # # # Calculate loss
            # # # Reshape for loss calculation: (batch*seq_len, vocab_size) vs (batch*seq_len)
            # # batch_size, seq_len, vocab_size = logits.shape
            # # logits_flat = logits.reshape(-1, vocab_size)
            # # targets_flat = tgt.reshape(-1)
            
            # # loss = self.criterion(logits_flat, targets_flat)
            
            # # # Backward pass with gradient clipping
            # # loss.backward()

            # # Calculate loss
            # batch_size, _, vocab_size = logits.shape
            # logits_flat = logits.reshape(-1, vocab_size)
            # targets_flat = targets_aligned.reshape(-1)
            
            # loss = self.criterion(logits_flat, targets_flat)

            # ---------------------------------------------------------
            # 1. AMP FORWARD PASS
            # ---------------------------------------------------------
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Model forward returns logits in fast 16-bit
                logits = self.model(src, tgt, src_mask)

                seq_len = logits.size(1)
                targets_aligned = tgt[:, -seq_len:]

                # Calculate loss
                batch_size, _, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = targets_aligned.reshape(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
            
            # Backward pass with gradient clipping
            self.scaler.scale(loss).backward()
            
            # # Gradient clipping (essential for LSTMs)
            # if self.config.gradient_clip_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(
            #         self.model.parameters(), 
            #         self.config.gradient_clip_norm
            #     )

            # ---------------------------------------------------------
            # 3. AMP GRADIENT CLIPPING
            # ---------------------------------------------------------
            # Gradient clipping (essential for LSTMs/Transformers)
            if self.config.gradient_clip_norm > 0:
                # You MUST unscale the gradients before clipping them!
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            # self.optimizer.step()

            # ---------------------------------------------------------
            # 4. AMP OPTIMIZER STEP
            # ---------------------------------------------------------
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- NEW: Step OneCycleLR per batch! ---
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            # ---------------------------------------
            
            # Update metrics
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                # metrics_calc.update(predictions, tgt, loss.item())
                metrics_calc.update(predictions, targets_aligned, loss.item())
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        return metrics_calc.compute()
    
    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        metrics_calc = MetricsCalculator(pad_idx=self.model.config.pad_idx)
        
        for batch in dataloader:
            # batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            src = batch['encoder_input'].to(self.device)
            tgt = batch['decoder_target'].to(self.device)
            # src_mask = batch.get('encoder_mask', None)
            src_mask = (src != self.model.config.pad_idx).to(self.device)
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            
            # Generate predictions (no teacher forcing)
            max_len = tgt.size(1)
            generated = self.model.generate(src, src_mask, max_len=max_len)
            
            # Pad or truncate to match target length for metric calculation
            if generated.size(1) < max_len:
                padding = torch.full(
                    (generated.size(0), max_len - generated.size(1)), 
                    self.model.config.pad_idx,
                    device=self.device
                )
                generated = torch.cat([generated, padding], dim=1)
            else:
                generated = generated[:, :max_len]
            
            # Calculate validation loss (with teacher forcing for consistency)
            logits = self.model(src, tgt, src_mask)
            # batch_size, seq_len, vocab_size = logits.shape
            # logits_flat = logits.reshape(-1, vocab_size)
            # targets_flat = tgt.reshape(-1)
            # loss = self.criterion(logits_flat, targets_flat).item()

            seq_len = logits.size(1)
            targets_aligned = tgt[:, -seq_len:]
            
            batch_size, _, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets_aligned.reshape(-1)
            
            loss = self.criterion(logits_flat, targets_flat).item()
            
            metrics_calc.update(generated, tgt, loss)
        
        return metrics_calc.compute()
    
    def fit(self, train_dataset, val_dataset=None):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (if None, uses subset of train)
        """
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create validation split if not provided
        if val_dataset is None:
            val_size = int(len(train_dataset) * self.config.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            logger.info(f"Split train into {train_size} train / {val_size} val")
        
        best_metric_value = float('-inf') if 'accuracy' in self.config.monitor_metric else float('inf')

        # best_metric_value = float('-inf') if 'accuracy' in self.config.monitor_metric else float('inf')
        
        # --- NEW: Initialize OneCycleLR for Transformers ---
        if self.config.lr_scheduler == "onecycle":
            # Estimate total steps based on the full dataset
            dummy_loader = DataLoader(train_dataset, batch_size=self.config.batch_size)
            total_steps = len(dummy_loader) * self.config.epochs
            
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1, # 10% warmup
                anneal_strategy='cos'
            )
            logger.info(f"Initialized OneCycleLR Warmup for {total_steps} total steps.")
        # ---------------------------------------------------
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Apply curriculum learning
            if self.curriculum:
                current_train_data = self._apply_curriculum(train_dataset, epoch)
            else:
                current_train_data = train_dataset
            
            # Create dataloaders
            train_loader = DataLoader(
                current_train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Train
            train_metrics = self._train_epoch(train_loader, epoch)
            self.train_metrics_history.append(train_metrics)
            
            # Validate
            if epoch % self.config.validate_every == 0:
                val_metrics = self._validate(val_loader)
                self.val_metrics_history.append(val_metrics)
                
                # # Learning rate scheduling
                # if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                #     self.scheduler.step(val_metrics[self.config.monitor_metric])
                # elif self.scheduler is not None:
                #     self.scheduler.step()

                # Learning rate scheduling
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config.monitor_metric])
                # --- MODIFIED: Prevent OneCycleLR from stepping here ---
                elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                
                # Check for improvement and checkpoint
                current_metric = val_metrics[self.config.monitor_metric]
                is_best = self.checkpoint_manager.check_improvement(current_metric)
                
                if is_best:
                    best_metric_value = current_metric
                
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, is_best=is_best
                )
                
                # Early stopping check
                if self.early_stopping(current_metric):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Logging
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Time: {time.time() - epoch_start_time:.1f}s | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val {self.config.monitor_metric}: {current_metric:.4f} | "
                    f"Val Seq Acc: {val_metrics['sequence_accuracy']:.4f} | "
                    f"Val BLEU: {val_metrics['bleu']:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch} (no validation) - Train Loss: {train_metrics['loss']:.4f}")
        
        logger.info("Training completed!")
        return self.train_metrics_history, self.val_metrics_history
    
    def save_history(self, path: str):
        """Save training history to JSON"""
        history = {
            'train': self.train_metrics_history,
            'val': self.val_metrics_history,
            'config': self.config.to_dict()
        }
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {path}")


# Convenience function for quick training setup
def train_model(model, train_dataset, val_dataset=None, 
                config: Optional[TrainingConfig] = None,
                checkpoint_dir: str = "/kaggle/working/checkpoints/"):
    """
    Quick training function with sensible defaults.
    
    Args:
        model: LSTM or Transformer model from Phase 3
        train_dataset: Training data
        val_dataset: Validation data (optional)
        config: Training configuration (uses defaults if None)
        checkpoint_dir: Where to save checkpoints
    
    Returns:
        Trainer instance with trained model
    """
    if config is None:
        config = TrainingConfig(
            checkpoint_dir=checkpoint_dir,
            use_curriculum=True,
            lr_scheduler="onecycle",
            early_stopping_patience=10
        )
    
    # Adjust gradient clip for LSTM specifically
    if hasattr(model, 'model_type') and model.model_type == 'lstm':
        config.gradient_clip_norm = 1.0  # Essential for LSTMs
        logger.info("LSTM detected: Using gradient clipping norm=1.0")
    
    trainer = Trainer(model, config)
    trainer.fit(train_dataset, val_dataset)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    from FASEROH.models.lstm_seq2seq import LSTMTaylorModel
    from FASEROH.models.transformer import TransformerTaylorModel
    from utils import ModelConfig
    
    # Create dummy model for testing
    model_config = ModelConfig(vocab_size=100, d_model=64, max_seq_len=50)
    model = LSTMTaylorModel(model_config)
    
    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=20):
            self.size = size
            self.seq_len = seq_len
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                'encoder_input': torch.randint(4, 100, (self.seq_len,)),
                'decoder_target': torch.randint(4, 100, (self.seq_len,)),
                'encoder_mask': torch.ones(self.seq_len, dtype=torch.bool)
            }
    
    train_data = DummyDataset(1000)
    val_data = DummyDataset(200)
    
    # Test training
    config = TrainingConfig(
        epochs=2,
        batch_size=512,
        checkpoint_dir="/kaggle/FASEROH/working/test_checkpoints/",
        use_curriculum=False,
        early_stopping_patience=5
    )
    
    trainer = Trainer(model, config)
    trainer.fit(train_data, val_data)
