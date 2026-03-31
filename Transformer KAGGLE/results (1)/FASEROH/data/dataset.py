
import json
import re
import pickle
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing import StrPrinter
import sys

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import sys
import os
# Dynamically find the project root instead of hardcoding /teamspace/
sys.path.append(os.getcwd())

# sys.path.append("/teamspace/studios/this_studio/")

from FASEROH.data.tokenizer import MathematicalTokenizer, TokenizerConfig

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class TaylorDatasetPipeline:
    """
    Complete data pipeline for Taylor expansion dataset.
    Handles loading, tokenization, batching, and masking.
    """
    
    def __init__(self, 
                 input_tokenizer: MathematicalTokenizer,
                 output_tokenizer: Optional[MathematicalTokenizer] = None,
                 max_input_len: int = 50,
                 max_output_len: int = 100):
        """
        Initialize pipeline.
        
        Args:
            input_tokenizer: Tokenizer for input functions
            output_tokenizer: Tokenizer for Taylor expansions (can be same as input)
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
        """
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer or input_tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        self.train_data: List[Dict] = []
        self.test_data: List[Dict] = []
        self.metadata: Dict = {}
    
    def load_from_jsonl(self, train_path: Path, test_path: Path):
        """Load dataset from JSONL files generated in Phase 1"""
        logger.info(f"Loading data from {train_path} and {test_path}")
        
        def load_jsonl(path):
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        
        self.train_data = load_jsonl(train_path)
        self.test_data = load_jsonl(test_path)
        
        logger.info(f"Loaded {len(self.train_data)} training samples, {len(self.test_data)} test samples")
        
        return self
    
    def prepare_sequence_pairs(self, split: str = 'train') -> Tuple[List[str], List[str]]:
        """
        Extract input-output string pairs from loaded data.
        
        Returns:
            Tuple of (input_expressions, target_expressions)
        """
        data = self.train_data if split == 'train' else self.test_data
        
        inputs = []
        targets = []
        
        for item in data:
            # Input: mathematical function
            inputs.append(item['input_func'])
            # Target: Taylor expansion
            targets.append(item['taylor_expansion'])
        
        return inputs, targets
    
    def create_tf_dataset(self, split: str = 'train', batch_size: int = 512):
        """
        Create TensorFlow dataset with proper masking.
        For use with Keras/TF models.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        inputs, targets = self.prepare_sequence_pairs(split)
        
        # Encode sequences
        encoder_input, encoder_mask = self.input_tokenizer.encode_batch(
            inputs, self.max_input_len
        )
        decoder_target, _ = self.output_tokenizer.encode_batch(
            targets, self.max_output_len
        )
        
        # Create decoder input (shifted right by 1, start with <START>)
        decoder_input = np.zeros_like(decoder_target)
        decoder_input[:, 0] = self.output_tokenizer.start_id
        decoder_input[:, 1:] = decoder_target[:, :-1]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'encoder_input': encoder_input,
            'encoder_mask': encoder_mask,
            'decoder_input': decoder_input,
            'decoder_target': decoder_target
        })
        
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        
        return dataset
    
    def create_pytorch_dataset(self, split: str = 'train'):
        """
        Create PyTorch dataset with padding masks.
        For use with PyTorch/Transformer models.
        """
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        inputs, targets = self.prepare_sequence_pairs(split)
        
        # Encode
        encoder_input, encoder_mask = self.input_tokenizer.encode_batch(
            inputs, self.max_input_len
        )
        decoder_target, decoder_mask = self.output_tokenizer.encode_batch(
            targets, self.max_output_len
        )
        
        # Create decoder input (teacher forcing)
        decoder_input = np.zeros_like(decoder_target)
        decoder_input[:, 0] = self.output_tokenizer.start_id
        decoder_input[:, 1:] = decoder_target[:, :-1]
        
        class TaylorDataset(Dataset):
            def __init__(self, enc_inp, enc_mask, dec_inp, dec_tgt):
                self.enc_inp = torch.LongTensor(enc_inp)
                self.enc_mask = torch.BoolTensor(enc_mask)
                self.dec_inp = torch.LongTensor(dec_inp)
                self.dec_tgt = torch.LongTensor(dec_tgt)
            
            def __len__(self):
                return len(self.enc_inp)
            
            def __getitem__(self, idx):
                return {
                    'encoder_input': self.enc_inp[idx],
                    'encoder_mask': self.enc_mask[idx],
                    'decoder_input': self.dec_inp[idx],
                    'decoder_target': self.dec_tgt[idx]
                }
        
        dataset = TaylorDataset(encoder_input, encoder_mask, decoder_input, decoder_target)
        
        return dataset
    
    def create_lytorch_dataloader(self, split: str = 'train', batch_size: int = 512, 
                                   shuffle: bool = True):
        """Convenience method to get PyTorch DataLoader"""
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("PyTorch is required")
        
        dataset = self.create_pytorch_dataset(split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def analyze_sequence_lengths(self) -> Dict:
        """Analyze length distribution to validate max_length choices"""
        inputs, targets = self.prepare_sequence_pairs('train')
        
        input_lengths = []
        target_lengths = []

        # --- ADDED TQDM PROGRESS BAR & DISABLED SYMPY ---
        logger.info("Analyzing sequence lengths...")
        for inp, tgt in tqdm(zip(inputs, targets), total=len(inputs), desc="Calculating Max Lengths"):
            # canonicalize=False stops the SymPy RAM freeze!
            inp_len = len(self.input_tokenizer.tokenize(inp, canonicalize=False))
            tgt_len = len(self.output_tokenizer.tokenize(tgt, canonicalize=False))
            input_lengths.append(inp_len)
            target_lengths.append(tgt_len)
        # ------------------------------------------------
        
        # for inp, tgt in zip(inputs, targets):
        #     inp_len = len(self.input_tokenizer.tokenize(inp))
        #     tgt_len = len(self.output_tokenizer.tokenize(tgt))
        #     input_lengths.append(inp_len)
        #     target_lengths.append(tgt_len)
        
        stats = {
            'input': {
                'mean': np.mean(input_lengths),
                'std': np.std(input_lengths),
                'max': np.max(input_lengths),
                'percentile_95': np.percentile(input_lengths, 95),
                'percentile_99': np.percentile(input_lengths, 99)
            },
            'target': {
                'mean': np.mean(target_lengths),
                'std': np.std(target_lengths),
                'max': np.max(target_lengths),
                'percentile_95': np.percentile(target_lengths, 95),
                'percentile_99': np.percentile(target_lengths, 99)
            }
        }
        
        logger.info("Sequence Length Analysis:")
        logger.info(f"Input - Mean: {stats['input']['mean']:.1f}, "
                   f"95th %ile: {stats['input']['percentile_95']:.1f}, "
                   f"Max: {stats['input']['max']}")
        logger.info(f"Target - Mean: {stats['target']['mean']:.1f}, "
                   f"95th %ile: {stats['target']['percentile_95']:.1f}, "
                   f"Max: {stats['target']['max']}")
        
        return stats

def build_complete_pipeline(train_jsonl: str, test_jsonl: str, 
                           output_dir: str = "/kaggle/working/tokenized_data"):
    """
    Complete pipeline: Load Phase 1 data, fit tokenizers, save processed datasets.
    
    Args:
        train_jsonl: Path to training JSONL file
        test_jsonl: Path to test JSONL file
        output_dir: Directory to save tokenized outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data to analyze
    train_data = []
    with open(train_jsonl, 'r') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    
    # Extract text
    input_exprs = [d['input_func'] for d in train_data]
    target_exprs = [d['taylor_expansion'] for d in train_data]
    
    # Create and fit input tokenizer
    logger.info("Fitting input tokenizer...")
    input_tokenizer = MathematicalTokenizer()
    input_tokenizer.fit(input_exprs, min_freq=1)
    input_tokenizer.save(output_dir / "input_tokenizer.pkl")
    
    # Create and fit output tokenizer (can share vocabulary with input)
    logger.info("Fitting output tokenizer...")
    output_tokenizer = MathematicalTokenizer()
    # Fit on both inputs and targets to ensure shared vocabulary
    all_exprs = input_exprs + target_exprs
    output_tokenizer.fit(all_exprs, min_freq=1)
    output_tokenizer.save(output_dir / "output_tokenizer.pkl")
    
    logger.info(f"Input vocab size: {input_tokenizer.vocab_size}")
    logger.info(f"Output vocab size: {output_tokenizer.vocab_size}")
    
    # Initialize pipeline
    pipeline = TaylorDatasetPipeline(
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        max_input_len=50,
        max_output_len=100
    )
    pipeline.load_from_jsonl(Path(train_jsonl), Path(test_jsonl))
    
    # Analyze lengths to validate max_length parameters
    stats = pipeline.analyze_sequence_lengths()
    
    # Adjust max lengths based on analysis if needed
    recommended_input_len = int(stats['input']['percentile_95']) + 2  # +2 for START/END
    recommended_output_len = int(stats['target']['percentile_95']) + 2
    
    logger.info(f"Recommended max_input_len: {recommended_input_len}")
    logger.info(f"Recommended max_output_len: {recommended_output_len}")
    
    # Save processed numpy arrays for fast loading
    logger.info("Processing and saving datasets...")
    
    for split in ['train', 'test']:
        inputs, targets = pipeline.prepare_sequence_pairs(split)
        
        # Encode
        enc_inp, enc_mask = input_tokenizer.encode_batch(inputs, max_length=50)
        dec_tgt, dec_mask = output_tokenizer.encode_batch(targets, max_length=100)
        
        # Create decoder input (shifted)
        dec_inp = np.zeros_like(dec_tgt)
        dec_inp[:, 0] = output_tokenizer.start_id
        dec_inp[:, 1:] = dec_tgt[:, :-1]
        
        # Save
        np.savez(
            output_dir / f"{split}_data.npz",
            encoder_input=enc_inp,
            encoder_mask=enc_mask,
            decoder_input=dec_inp,
            decoder_target=dec_tgt,
            decoder_mask=dec_mask
        )
        
        logger.info(f"Saved {split} set: {len(inputs)} samples")
    
    # Save metadata
    metadata = {
        'vocab_size_input': input_tokenizer.vocab_size,
        'vocab_size_output': output_tokenizer.vocab_size,
        'max_input_length': 50,
        'max_output_length': 100,
        'train_samples': len(pipeline.train_data),
        'test_samples': len(pipeline.test_data),
        'sequence_stats': stats,
        'special_tokens': {
            'pad': input_tokenizer.pad_id,
            'unk': input_tokenizer.unk_id,
            'start': input_tokenizer.start_id,
            'end': input_tokenizer.end_id
        }
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Pipeline complete. Data saved to {output_dir}")
    
    return pipeline, metadata

# Example usage and testing
if __name__ == "__main__":
    # Example: Test tokenization
    tokenizer = MathematicalTokenizer()
    
    test_exprs = [
        "sin(x) + x**2",
        "exp(-x)*cos(2*x)",
        "x*log(1+x**2)",
        "1 + x + x**2/2 - x**3/6"
    ]
    
    print("Testing Tokenization:")
    print("=" * 60)
    
    for expr in test_exprs:
        tokens = tokenizer.tokenize(expr, canonicalize=True)
        print(f"\nInput:  {expr}")
        print(f"Tokens: {tokens}")
        
        # Simulate encoding (need to fit first for real IDs)
        if tokenizer.is_fitted:
            ids = tokenizer.encode(expr)
            decoded = tokenizer.decode(ids)
            print(f"IDs:    {ids}")
            print(f"Back:   {decoded}")
    
    # Fit on test data
    tokenizer.fit(test_exprs)
    
    print("\n" + "=" * 60)
    print("After Fitting:")
    print("=" * 60)
    
    for expr in test_exprs:
        tokens = tokenizer.tokenize(expr)
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        print(f"\nInput:  {expr}")
        print(f"Tokens: {tokens}")
        print(f"IDs:    {ids}")
        print(f"Back:   {decoded}")
    
    # Test batch encoding
    padded, masks = tokenizer.encode_batch(test_exprs, max_length=20)
    print(f"\nBatch shape: {padded.shape}")
    print(f"Mask shape:  {masks.shape}")
    print(f"First sample IDs: {padded[0]}")
    print(f"First sample mask: {masks[0]}")
