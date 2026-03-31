
import re
import pickle
from typing import List, Dict, Tuple, Optional, Union, Set, Iterable
from dataclasses import dataclass, field, asdict
import sympy as sp
from sympy.printing import StrPrinter
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from collections import Counter
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Configuration for mathematical expression tokenizer"""
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    
    # Multi-character mathematical tokens (domain-specific)
    # This reduces sequence length and captures mathematical semantics better than pure char-level
    math_tokens: List[str] = field(default_factory=lambda: [
        "sin", "cos", "tan", "exp", "log", "sqrt",  # Functions
        "**",  # Power operator (Python syntax)
        "pi", "E",  # Constants
    ])
    
    # Character set for basic tokenization
    allowed_chars: Set[str] = field(default_factory=lambda: set(
        "x0123456789+-*/()^.,= "
    ))
    
    max_input_length: int = 50
    max_output_length: int = 100
    
    # SymPy canonicalization settings
    expand_expression: bool = True
    simplify_rational: bool = True
    sort_terms: bool = True  # Ascending powers

class MathematicalTokenizer:
    """
    Intelligent tokenizer for mathematical expressions.
    Uses hybrid approach: multi-char tokens for functions, char-level for rest.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        
        # Vocabulary mappings
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        # Special token IDs
        self.pad_id: int = 0
        self.unk_id: int = 1
        self.start_id: int = 2
        self.end_id: int = 3
        
        # Compiled regex patterns for tokenization
        self._compile_patterns()
        
        self.is_fitted = False
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient tokenization"""
        # Sort math tokens by length (descending) to match longest first
        sorted_math_tokens = sorted(self.config.math_tokens, key=len, reverse=True)
        
        # Build pattern: match multi-char tokens OR single allowed chars
        math_pattern = '|'.join(re.escape(tok) for tok in sorted_math_tokens)
        char_pattern = '[' + re.escape(''.join(self.config.allowed_chars)) + ']'
        
        # Combined pattern: either math token or single character
        self.token_pattern = re.compile(f'({math_pattern}|{char_pattern})')
        
        # Pattern to identify numbers (including decimals and negatives)
        self.number_pattern = re.compile(r'-?\d+\.?\d*')
    
    def _canonicalize_expression(self, expr_str: str) -> str:
        """
        Canonicalize mathematical expression using SymPy.
        Ensures consistent formatting: ascending powers, simplified fractions.
        """
        try:
            # Parse string to SymPy expression
            x = sp.Symbol('x')
            # Replace ^ with ** for Python compatibility if needed
            expr_str = expr_str.replace('^', '**')
            
            expr = parse_expr(expr_str, local_dict={'x': x, 'sin': sp.sin, 'cos': sp.cos, 
                                                     'exp': sp.exp, 'log': sp.log, 
                                                     'sqrt': sp.sqrt, 'tan': sp.tan})
            
            # Expand expression (distribute products over sums)
            if self.config.expand_expression:
                expr = sp.expand(expr)
            
            # Simplify rational numbers
            if self.config.simplify_rational:
                expr = sp.nsimplify(expr, [sp.pi, sp.E])
                expr = sp.simplify(expr)
            
            # Convert back to string with consistent formatting
            # result = str(expr)

            printer = StrPrinter({'order': 'rev-lex'})
            result = printer.doprint(expr)
            
            # Normalize whitespace
            result = ' '.join(result.split())
            
            return result
            
        except Exception as e:
            logger.warning(f"Canonicalization failed for '{expr_str}': {e}")
            # Return cleaned original if parsing fails
            return expr_str.replace('^', '**').strip()
    
    def tokenize(self, expression: str, canonicalize: bool = True) -> List[str]:
        """
        Tokenize mathematical expression into list of tokens.
        
        Args:
            expression: Mathematical expression string
            canonicalize: Whether to canonicalize using SymPy first
            
        Returns:
            List of tokens
        """
        if canonicalize:
            expression = self._canonicalize_expression(expression)
        
        # Insert spaces around operators to ensure proper splitting
        # But preserve multi-char operators like **
        expr_processed = expression
        
        # Find all tokens using regex
        tokens = self.token_pattern.findall(expr_processed)
        
        # Filter out empty strings and whitespace-only tokens
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return tokens
    
    # def fit(self, expressions: List[str], min_freq: int = 1):
    #     """
    #     Build vocabulary from list of expressions.
        
    #     Args:
    #         expressions: List of mathematical expression strings
    #         min_freq: Minimum frequency for token to be included
    #     """
    #     logger.info(f"Building vocabulary from {len(expressions)} expressions...")
        
    #     # Count token frequencies
    #     token_counter = Counter()
        
    #     for expr in expressions:
    #         tokens = self.tokenize(expr, canonicalize=True)
    #         token_counter.update(tokens)
        
    #     # Initialize vocabulary with special tokens
    #     self.token_to_idx = {
    #         self.config.pad_token: self.pad_id,
    #         self.config.unk_token: self.unk_id,
    #         self.config.start_token: self.start_id,
    #         self.config.end_token: self.end_id,
    #     }
        
    #     # Add frequent tokens to vocabulary
    #     idx = len(self.token_to_idx)
    #     for token, freq in token_counter.most_common():
    #         if freq >= min_freq and token not in self.token_to_idx:
    #             self.token_to_idx[token] = idx
    #             idx += 1
        
    #     # Create reverse mapping
    #     self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
    #     self.vocab_size = len(self.token_to_idx)
    #     self.is_fitted = True
        
    #     logger.info(f"Vocabulary built: {self.vocab_size} tokens")
    #     logger.info(f"Top 10 tokens: {token_counter.most_common(10)}")
        
    #     return self

    def fit(self, expressions: Iterable[str], min_freq: int = 1):
        """
        Build vocabulary from an iterable of expressions (supports streaming).
        
        Args:
            expressions: Iterable/Generator of mathematical expression strings
            min_freq: Minimum frequency for token to be included
        """
        logger.info("Building vocabulary (streaming mode)...")
        
        # Count token frequencies
        token_counter = Counter()
        
        # Stream processing: Only holds one equation in RAM at a time
        for i, expr in enumerate(expressions):
            # --- THE FIX: canonicalize=False ---
            # SymPy caches parsed trees. Re-parsing 65k strings causes an OOM crash.
            # The Phase 1 data generator already formatted these perfectly!
            tokens = self.tokenize(expr, canonicalize=False)
            token_counter.update(tokens)
            
            # Print a heartbeat so you know it hasn't frozen!
            if (i + 1) % 10000 == 0:
                logger.info(f"  Scanned {i + 1} expressions...")
        
        # Initialize vocabulary with special tokens
        self.token_to_idx = {
            self.config.pad_token: self.pad_id,
            self.config.unk_token: self.unk_id,
            self.config.start_token: self.start_id,
            self.config.end_token: self.end_id,
        }
        
        # Add frequent tokens to vocabulary
        idx = len(self.token_to_idx)
        for token, freq in token_counter.most_common():
            if freq >= min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = idx
                idx += 1
        
        # Create reverse mapping
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)
        self.is_fitted = True
        
        logger.info(f"Vocabulary built: {self.vocab_size} tokens")
        logger.info(f"Top 10 tokens: {token_counter.most_common(10)}")
        
        return self
    
    def encode(self, expression: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode expression to list of integer indices.
        
        Args:
            expression: Mathematical expression string
            add_special_tokens: Whether to add <START> and <END>
            max_length: Maximum sequence length (truncate if longer)
            
        Returns:
            List of integer token IDs
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding")
        
        tokens = self.tokenize(expression, canonicalize = False)
        
        # Convert to IDs
        token_ids = [self.token_to_idx.get(tok, self.unk_id) for tok in tokens]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.start_id] + token_ids + [self.end_id]
        
        # Truncate if necessary
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.end_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode list of integer indices back to string.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove <PAD>, <START>, <END>, <UNK>
            
        Returns:
            Reconstructed expression string
        """
        tokens = []
        for idx in token_ids:
            token = self.idx_to_token.get(idx, self.config.unk_token)
            
            if skip_special_tokens and token in [
                self.config.pad_token, self.config.start_token, 
                self.config.end_token, self.config.unk_token
            ]:
                continue
            
            tokens.append(token)
        
        # Join with spaces and cleanup
        result = ''.join(tokens)
        # Fix spacing around operators for readability
        result = result.replace('**', '^')  # Convert back to caret for readability if desired
        
        return result
    
    def encode_batch(self, expressions: List[str], max_length: int,
                     pad_to_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode batch of expressions with padding and masking.
        
        Returns:
            Tuple of (padded_sequences, attention_masks)
            padded_sequences: (batch_size, max_length) array of token IDs
            attention_masks: (batch_size, max_length) binary array (1=real token, 0=pad)
        """
        sequences = []
        lengths = []
        
        # for expr in expressions:
        #     seq = self.encode(expr, add_special_tokens=True, max_length=max_length)
        #     sequences.append(seq)
        #     lengths.append(len(seq))

        # --- ADDED TQDM TO THE SILENT ENCODING LOOP ---
        for expr in tqdm(expressions, desc="Encoding Batch to Tensors"):
            seq = self.encode(expr, add_special_tokens=True, max_length=max_length)
            sequences.append(seq)
            lengths.append(len(seq))
        # ----------------------------------------------
        
        # Padding
        if pad_to_max:
            padded = np.full((len(sequences), max_length), self.pad_id, dtype=np.int32)
            masks = np.zeros((len(sequences), max_length), dtype=np.int32)
            
            for i, seq in enumerate(sequences):
                end_idx = min(len(seq), max_length)
                padded[i, :end_idx] = seq[:end_idx]
                masks[i, :end_idx] = 1  # 1 for real tokens, 0 for pad
        
        return padded, masks
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer vocabulary and config"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'config': {
                'pad_token': self.config.pad_token,
                'unk_token': self.config.unk_token,
                'start_token': self.config.start_token,
                'end_token': self.config.end_token,
                'math_tokens': self.config.math_tokens,
                'allowed_chars': list(self.config.allowed_chars),
                'max_input_length': self.config.max_input_length,
                'max_output_length': self.config.max_output_length,
            },
            'vocab_size': self.vocab_size,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MathematicalTokenizer':
        """Load tokenizer from disk"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Reconstruct config
        config = TokenizerConfig(
            pad_token=state['config']['pad_token'],
            unk_token=state['config']['unk_token'],
            start_token=state['config']['start_token'],
            end_token=state['config']['end_token'],
            math_tokens=state['config']['math_tokens'],
            allowed_chars=set(state['config']['allowed_chars']),
            max_input_length=state['config']['max_input_length'],
            max_output_length=state['config']['max_output_length'],
        )
        
        tokenizer = cls(config)
        tokenizer.token_to_idx = state['token_to_idx']
        tokenizer.idx_to_token = {int(k): v for k, v in state['idx_to_token'].items()}
        tokenizer.vocab_size = state['vocab_size']
        tokenizer.is_fitted = state['is_fitted']
        tokenizer._compile_patterns()
        
        return tokenizer
