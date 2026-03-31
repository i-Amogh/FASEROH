
import torch
import numpy as np
import json
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import logging
from tqdm import tqdm
import warnings

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for comprehensive evaluation results"""
    # Basic metrics
    token_accuracy: float = 0.0
    sequence_accuracy: float = 0.0
    bleu_score: float = 0.0
    perplexity: float = 0.0
    
    # Mathematical correctness (symbolic)
    symbolic_accuracy: float = 0.0
    symbolic_equivalence_rate: float = 0.0  # Syntactically different but mathematically equal
    
    # Categorized metrics
    by_complexity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_base_function: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Error analysis
    total_samples: int = 0
    correct_predictions: int = 0
    symbolic_matches: int = 0
    equivalence_matches: int = 0
    
    # Detailed results for visualization
    predictions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'token_accuracy': self.token_accuracy,
            'sequence_accuracy': self.sequence_accuracy,
            'bleu_score': self.bleu_score,
            'perplexity': self.perplexity,
            'symbolic_accuracy': self.symbolic_accuracy,
            'symbolic_equivalence_rate': self.symbolic_equivalence_rate,
            'by_complexity': self.by_complexity,
            'by_base_function': self.by_base_function,
            'total_samples': self.total_samples,
            'correct_predictions': self.correct_predictions,
            'symbolic_matches': self.symbolic_matches,
        }


class SymbolicVerifier:
    """
    Verifies mathematical equivalence using SymPy.
    Handles cases like 1/2*x vs x/2 that string comparison misses.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.x = sp.Symbol('x')
        self.tolerance = tolerance
        
    def parse_expression(self, expr_str: str) -> Optional[sp.Expr]:
        """Safely parse string to SymPy expression"""
        try:
            # Clean up common formatting issues
            expr_str = expr_str.strip()
            expr_str = expr_str.replace('^', '**')
            expr_str = expr_str.replace('<START>', '').replace('<END>', '').replace('<PAD>', '')
            expr_str = expr_str.replace('<UNK>', '')
            
            # Handle implicit multiplication (e.g., "2x" -> "2*x")
            # This is a simplified version; real implementation would need more robust parsing
            expr_str = expr_str.replace(')(', ')*(')
            
            return parse_expr(expr_str, local_dict={'x': self.x, 'X': self.x})
        except Exception as e:
            logger.debug(f"Failed to parse '{expr_str}': {e}")
            return None
    
    def check_equivalence(self, pred_str: str, truth_str: str) -> Tuple[bool, bool]:
        """
        Check mathematical equivalence between two expressions.
        
        Returns:
            (exact_match, symbolic_equivalence)
            exact_match: Syntactically identical (after normalization)
            symbolic_equivalence: Mathematically equal (simplify(pred - truth) == 0)
        """
        # String-level exact match (after basic normalization)
        pred_normalized = ' '.join(pred_str.lower().split())
        truth_normalized = ' '.join(truth_str.lower().split())
        exact_match = pred_normalized == truth_normalized
        
        # Symbolic verification
        pred_expr = self.parse_expression(pred_str)
        truth_expr = self.parse_expression(truth_str)
        
        if pred_expr is None or truth_expr is None:
            return exact_match, False
        
        try:
            # Check difference is zero
            diff = sp.simplify(pred_expr - truth_expr)
            symbolic_match = diff == 0
            
            # Additional numerical verification for safety
            if not symbolic_match:
                # Test at several points
                test_points = [0.1, 0.5, 1.0, -0.1, -0.5]
                numerical_match = True
                for pt in test_points:
                    try:
                        pred_val = float(pred_expr.subs(self.x, pt).evalf())
                        truth_val = float(truth_expr.subs(self.x, pt).evalf())
                        if abs(pred_val - truth_val) > self.tolerance:
                            numerical_match = False
                            break
                    except:
                        numerical_match = False
                        break
                
                # If numerically matches but not symbolically, it's an equivalent form
                equivalent_form = numerical_match and not symbolic_match
            else:
                equivalent_form = False
            
            return exact_match, (symbolic_match or equivalent_form)
            
        except Exception as e:
            logger.debug(f"Symbolic comparison failed: {e}")
            return exact_match, False


class AttentionVisualizer:
    """
    Generates attention heatmaps for LSTM and Transformer models.
    Shows alignment between input and output tokens.
    """
    
    def __init__(self, tokenizer, save_dir: str = "/kaggle/working/plots/"):
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    # def extract_attention_weights(self, model, batch: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
    #     """Extract attention weights from model during forward pass"""
    #     model.eval()
        
    #     with torch.no_grad():
            
    #         src = batch['encoder_input']
    #         tgt = batch.get('decoder_input', batch.get('decoder_target'))
            
    #         # For LSTM with attention mechanism
    #         if hasattr(model, 'decoder') and hasattr(model.decoder, 'enable_attention_storage'):
    #             model.decoder.enable_attention_storage(True)
    #             _ = model(src, tgt)
    #             attn_matrix = model.decoder.get_attention_matrix()
    #             model.decoder.enable_attention_storage(False)
    #             return attn_matrix
            
    #         # For Transformer, extract from last decoder layer
    #         elif hasattr(model, 'decoder_layers'):
    #             model.eval()
    #             with torch.no_grad():
    #             # Pass dummy target to get the cross-attention weights
    #                 src = batch['encoder_input']
    #                 tgt = batch['decoder_target']

    #                 device = next(model.parameters()).device
    #                 src = src.to(device)
    #                 tgt = tgt.to(device)
        
    #                 # Ensure your model's forward returns (logits, attn_weights)
    #                 _, attn_weights = model(src, tgt, return_attention=True) 
    #                 return attn_weights.cpu().numpy()
            
    #         else:
    #             return None

    def extract_attention_weights(self, model, batch: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
        """Extract attention weights from model during forward pass"""
        model.eval()
        
        with torch.no_grad():
            # 1. Dynamically find the model's device
            device = next(model.parameters()).device
            
            # 2. Extract and immediately push to GPU
            src = batch['encoder_input'].to(device)
            
            # Use get() safely, then push to device if it exists
            tgt_raw = batch.get('decoder_input', batch.get('decoder_target'))
            tgt = tgt_raw.to(device) if tgt_raw is not None else None
            
            # ---------------------------------------------------------
            # For LSTM with attention mechanism
            # ---------------------------------------------------------
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'enable_attention_storage'):
                model.decoder.enable_attention_storage(True)
                
                # src and tgt are now safely on cuda:0
                _ = model(src, tgt) 
                
                attn_matrix = model.decoder.get_attention_matrix()
                model.decoder.enable_attention_storage(False)
                
                # Ensure we return it as a numpy array for visualization
                if isinstance(attn_matrix, torch.Tensor):
                    return attn_matrix.cpu().numpy()
                return attn_matrix
            
            # ---------------------------------------------------------
            # For Transformer, extract from last decoder layer
            # ---------------------------------------------------------
            elif hasattr(model, 'decoder_layers'):
                # src and tgt are already on cuda:0 from step 2
                
                # Ensure your model's forward returns (logits, attn_weights)
                _, attn_weights = model(src, tgt, return_attention=True) 
                return attn_weights.cpu().numpy()
            
            else:
                return None
    
    def create_heatmap(self, attention: np.ndarray, 
                       src_tokens: List[str], 
                       tgt_tokens: List[str],
                       title: str = "Attention Alignment",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            attention: (tgt_len, src_len) attention weights
            src_tokens: Source token strings
            tgt_tokens: Target token strings
            title: Plot title
            save_path: Where to save figure
        """
        fig, ax = plt.subplots(figsize=(max(8, len(src_tokens) * 0.5), 
                                      max(6, len(tgt_tokens) * 0.4)))
        
        # Create heatmap
        sns.heatmap(
            attention,
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cmap='viridis',
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            square=True
        )
        
        ax.set_xlabel('Input Function Tokens', fontsize=12)
        ax.set_ylabel('Taylor Expansion Tokens', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention visualization to {save_path}")
        
        return fig
    
    def visualize_sample(self, model, batch: Dict, idx: int = 0,
                        save_name: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Visualize attention for a single sample in batch.
        
        Args:
            model: The seq2seq model
            batch: Data batch
            idx: Index in batch to visualize
            save_name: Filename for saving
        """
        attention = self.extract_attention_weights(model, batch)
        
        if attention is None:
            logger.warning("Could not extract attention weights from model")
            return None
        
        # Get single sample attention
        if len(attention.shape) == 3:  # (batch, tgt_len, src_len)
            sample_attn = attention[idx]
        else:
            sample_attn = attention
        
        # Decode tokens
        src_ids = batch['encoder_input'][idx].cpu().numpy()
        tgt_ids = batch.get('decoder_input', batch.get('decoder_target'))[idx].cpu().numpy()
        
        src_tokens = [self.tokenizer.decode([t], skip_special_tokens=False) for t in src_ids]
        tgt_tokens = [self.tokenizer.decode([t], skip_special_tokens=False) for t in tgt_ids]
        
        # Clean up tokens for display
        src_tokens = [t for t in src_tokens if t not in ['<PAD>', '']]
        tgt_tokens = [t for t in tgt_tokens if t not in ['<PAD>', '']]
        
        # Truncate attention matrix to match cleaned tokens
        sample_attn = sample_attn[:len(tgt_tokens), :len(src_tokens)]
        
        title = f"Attention: {' '.join(src_tokens[:10])}..."
        if save_name is None:
            save_name = f"attention_sample_{idx}.png"
        
        save_path = self.save_dir / save_name
        
        return self.create_heatmap(sample_attn, src_tokens, tgt_tokens, title, str(save_path))


class CategorizedEvaluator:
    """
    Evaluates model performance broken down by categories.
    Identifies strengths and weaknesses across function types.
    """
    
    def __init__(self, symbolic_verifier: SymbolicVerifier):
        self.verifier = symbolic_verifier
        self.results_by_category = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'symbolic_correct': 0,
            'token_correct': 0,
            'token_total': 0
        })
    
    def categorize_sample(self, sample: Dict) -> Tuple[str, List[str]]:
        """
        Determine categories for a sample.
        Returns (complexity_class, base_functions).
        """
        complexity = sample.get('complexity_class', 'unknown')
        base_funcs = sample.get('base_functions', [])
        
        # Normalize base functions
        if isinstance(base_funcs, str):
            base_funcs = [base_funcs]
        
        return complexity, base_funcs
    
    def update(self, sample: Dict, pred_str: str, truth_str: str, 
               token_acc: float, exact_match: bool, symbolic_match: bool):
        """Update category statistics"""
        complexity, base_funcs = self.categorize_sample(sample)
        
        # Update by complexity
        cat = self.results_by_category[complexity]
        cat['total'] += 1
        cat['correct'] += int(exact_match)
        cat['symbolic_correct'] += int(symbolic_match)
        cat['token_correct'] += int(token_acc * 100)  # Store as percentage sum
        cat['token_total'] += 1
        
        # Update by base function
        for func in base_funcs:
            cat = self.results_by_category[f"func_{func}"]
            cat['total'] += 1
            cat['correct'] += int(exact_match)
            cat['symbolic_correct'] += int(symbolic_match)
            cat['token_correct'] += int(token_acc * 100)
            cat['token_total'] += 1
    
    def get_results(self) -> Dict[str, Dict[str, float]]:
        """Calculate final metrics by category"""
        results = {}
        
        for category, stats in self.results_by_category.items():
            if stats['total'] > 0:
                results[category] = {
                    'sequence_accuracy': stats['correct'] / stats['total'],
                    'symbolic_accuracy': stats['symbolic_correct'] / stats['total'],
                    'token_accuracy': (stats['token_correct'] / stats['token_total']) / 100 if stats['token_total'] > 0 else 0,
                    'count': stats['total']
                }
        
        return results


class ModelComparator:
    """
    Compare multiple models side-by-side.
    Generates comparison tables and convergence graphs.
    """
    
    def __init__(self, save_dir: str = "/kaggle/working/comparisons/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.histories = {}
        self.evaluations = {}
    
    def add_model(self, name: str, history: Dict, evaluation: EvaluationResult,
                  training_time: Optional[float] = None):
        """Add model results for comparison"""
        self.histories[name] = history
        self.evaluations[name] = {
            'result': evaluation,
            'training_time': training_time
        }
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create side-by-side comparison DataFrame"""
        data = []
        
        for name, eval_data in self.evaluations.items():
            result = eval_data['result']
            row = {
                'Model': name,
                'Token Accuracy': f"{result.token_accuracy:.4f}",
                'Sequence Accuracy': f"{result.sequence_accuracy:.4f}",
                'Symbolic Accuracy': f"{result.symbolic_accuracy:.4f}",
                'BLEU Score': f"{result.bleu_score:.4f}",
                'Perplexity': f"{result.perplexity:.2f}",
                'Training Time (s)': f"{eval_data.get('training_time', 0):.1f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def plot_convergence(self, metric: str = 'sequence_accuracy', 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training convergence for all models.
        
        Args:
            metric: Which metric to plot ('loss', 'token_accuracy', 'sequence_accuracy')
            save_path: Where to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for name, history in self.histories.items():
            # Training curves
            if 'train' in history and len(history['train']) > 0:
                train_vals = [h.get(metric, 0) for h in history['train']]
                epochs = range(1, len(train_vals) + 1)
                ax1.plot(epochs, train_vals, label=f'{name} (train)', linestyle='--')
            
            # Validation curves
            if 'val' in history and len(history['val']) > 0:
                val_vals = [h.get(metric, 0) for h in history['val']]
                epochs = range(1, len(val_vals) + 1)
                ax2.plot(epochs, val_vals, label=f'{name} (val)', marker='o')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title('Training Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title('Validation Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_category_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Bar chart comparing models by complexity category"""
        categories = set()
        for eval_data in self.evaluations.values():
            categories.update(eval_data['result'].by_complexity.keys())
        
        categories = sorted(categories)
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (name, eval_data) in enumerate(self.evaluations.items()):
            result = eval_data['result']
            values = [result.by_complexity.get(cat, {}).get('sequence_accuracy', 0) 
                     for cat in categories]
            offset = width * (i - len(self.evaluations)/2 + 0.5)
            ax.bar(x + offset, values, width, label=name)
        
        ax.set_xlabel('Complexity Category')
        ax.set_ylabel('Sequence Accuracy')
        ax.set_title('Model Comparison by Function Complexity')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, output_file: str = "comparison_report.md"):
        """Generate comprehensive markdown report"""
        report_path = self.save_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            
            # Summary table
            f.write("## Overall Performance\n\n")
            df = self.create_comparison_table()
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # Detailed breakdown
            f.write("## Performance by Complexity\n\n")
            for name, eval_data in self.evaluations.items():
                f.write(f"### {name}\n\n")
                result = eval_data['result']
                
                f.write("**By Complexity:**\n")
                for cat, metrics in result.by_complexity.items():
                    f.write(f"- {cat}: Seq Acc={metrics['sequence_accuracy']:.3f}, "
                           f"Sym Acc={metrics['symbolic_accuracy']:.3f} "
                           f"(n={metrics['count']})\n")
                
                f.write("\n**By Base Function:**\n")
                for cat, metrics in result.by_base_function.items():
                    if cat.startswith('func_'):
                        func_name = cat[5:]
                        f.write(f"- {func_name}: Seq Acc={metrics['sequence_accuracy']:.3f} "
                               f"(n={metrics['count']})\n")
                f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            best_model = max(self.evaluations.items(), 
                           key=lambda x: x[1]['result'].symbolic_accuracy)
            f.write(f"- **Best Overall Model**: {best_model[0]} "
                   f"(Symbolic Accuracy: {best_model[1]['result'].symbolic_accuracy:.3f})\n")
            
            # Identify weaknesses
            for name, eval_data in self.evaluations.items():
                result = eval_data['result']
                weakest = min(result.by_complexity.items(), 
                             key=lambda x: x[1]['sequence_accuracy'])
                f.write(f"- **{name}** struggles most with **{weakest[0]}** functions "
                       f"(accuracy: {weakest[1]['sequence_accuracy']:.3f})\n")
        
        logger.info(f"Generated comparison report at {report_path}")


class AblationStudy:
    """
    Conduct ablation studies to quantify impact of specific components.
    """
    
    def __init__(self, base_config, trainer_class):
        self.base_config = base_config
        self.trainer_class = trainer_class
        self.results = {}
    
    def run_attention_ablation(self, train_data, val_data, test_data):
        """
        Compare LSTM with and without attention mechanism.
        """
        from models import LSTMTaylorModel, ModelConfig
        
        results = {}
        
        # With attention (Bahdanau)
        logger.info("Training LSTM with Bahdanau attention...")
        config_with = self.base_config.copy()
        config_with['attention_type'] = 'bahdanau'
        model_with = LSTMTaylorModel(ModelConfig(**config_with))
        # Train and evaluate...
        
        # Without attention (simple decoder)
        logger.info("Training LSTM without attention...")
        # Would require modifying model to disable attention
        # For now, document the interface
        
        return results
    
    def run_positional_encoding_ablation(self, train_data, val_data, test_data):
        """
        Compare Transformer with sinusoidal vs learned positional encoding.
        """
        from models import TransformerTaylorModel, ModelConfig
        
        results = {}
        
        for pe_type in ['sinusoidal', 'learned']:
            logger.info(f"Training Transformer with {pe_type} PE...")
            config = self.base_config.copy()
            config['positional_encoding'] = pe_type
            model = TransformerTaylorModel(ModelConfig(**config))
            # Train and evaluate...
            results[pe_type] = None  # Store evaluation result
        
        return results


class TaylorEvaluator:
    """
    Main evaluation orchestrator.
    Combines all evaluation components.
    """
    
    def __init__(self, model, tokenizer, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if device != 'auto' else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()
        
        self.verifier = SymbolicVerifier()
        self.visualizer = AttentionVisualizer(tokenizer)
        self.categorized = CategorizedEvaluator(self.verifier)
    
    @torch.no_grad()
    def evaluate(self, test_dataset, batch_size: int = 512, 
                 max_samples: Optional[int] = None) -> EvaluationResult:
        """
        Comprehensive evaluation on test set.
        """
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        result = EvaluationResult()
        all_predictions = []
        
        total_tokens = 0
        correct_tokens = 0
        
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            if max_samples and batch_idx * batch_size >= max_samples:
                break
            
            # Move to device
            src = batch['encoder_input'].to(self.device)
            tgt = batch['decoder_target'].to(self.device)
            src_mask = batch.get('encoder_mask')
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            
            # Generate predictions
            max_len = tgt.size(1)
            predictions = self.model.generate(src, src_mask, max_len=max_len)
            
            # Process each sample in batch
            for i in range(src.size(0)):
                # Decode predictions and targets
                pred_ids = predictions[i].cpu().numpy()
                tgt_ids = tgt[i].cpu().numpy()
                
                # Remove padding and special tokens
                pred_str = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                truth_str = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                
                # Token-level accuracy
                min_len = min(len(pred_ids), len(tgt_ids))
                token_matches = (pred_ids[:min_len] == tgt_ids[:min_len]).sum()
                token_total = len([t for t in tgt_ids if t != self.tokenizer.pad_id])
                
                token_acc = token_matches / max(token_total, 1)
                correct_tokens += token_matches
                total_tokens += token_total
                
                # Exact match
                exact_match = pred_str.strip() == truth_str.strip()
                
                # Symbolic verification
                exact_sym, symbolic_match = self.verifier.check_equivalence(pred_str, truth_str)
                
                # Update results
                result.total_samples += 1
                result.correct_predictions += int(exact_match)
                result.symbolic_matches += int(symbolic_match)
                result.equivalence_matches += int(symbolic_match and not exact_match)
                
                # Get sample metadata if available
                sample_meta = {
                    'complexity_class': batch.get('complexity_class', ['unknown']*src.size(0))[i] if isinstance(batch.get('complexity_class'), list) else 'unknown',
                    'base_functions': batch.get('base_functions', [[]]*src.size(0))[i] if isinstance(batch.get('base_functions'), list) else []
                }
                
                # Update categorized results
                self.categorized.update(sample_meta, pred_str, truth_str, 
                                       token_acc, exact_match, symbolic_match)
                
                # Store detailed result
                all_predictions.append({
                    'input': self.tokenizer.decode(src[i].cpu().numpy(), skip_special_tokens=True),
                    'predicted': pred_str,
                    'truth': truth_str,
                    'exact_match': exact_match,
                    'symbolic_match': symbolic_match,
                    'token_accuracy': token_acc,
                    'complexity': sample_meta['complexity_class']
                })
        
        # Calculate final metrics
        result.token_accuracy = correct_tokens / max(total_tokens, 1)
        result.sequence_accuracy = result.correct_predictions / max(result.total_samples, 1)
        result.symbolic_accuracy = result.symbolic_matches / max(result.total_samples, 1)
        result.symbolic_equivalence_rate = result.equivalence_matches / max(result.total_samples, 1)
        result.by_complexity = self.categorized.get_results()
        result.predictions = all_predictions
        
        # Log summary
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Samples: {result.total_samples}")
        logger.info(f"Token Accuracy: {result.token_accuracy:.4f}")
        logger.info(f"Sequence Accuracy: {result.sequence_accuracy:.4f}")
        logger.info(f"Symbolic Accuracy: {result.symbolic_accuracy:.4f}")
        logger.info(f"Equivalent Forms (diff syntax, same math): {result.symbolic_equivalence_rate:.4f}")
        
        logger.info("\nBy Complexity:")
        for cat, metrics in result.by_complexity.items():
            if not cat.startswith('func_'):
                logger.info(f"  {cat:15s}: SeqAcc={metrics['sequence_accuracy']:.3f}, "
                           f"SymAcc={metrics['symbolic_accuracy']:.3f}, "
                           f"n={metrics['count']}")
        
        return result
    
    def visualize_attention(self, test_dataset, num_samples: int = 5):
        """Generate attention visualizations for samples"""
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        batch = next(iter(dataloader))
        
        for i in range(min(num_samples, batch['encoder_input'].size(0))):
            self.visualizer.visualize_sample(
                self.model, batch, idx=i,
                save_name=f"attention_sample_{i}.png"
            )
    
    def save_results(self, result: EvaluationResult, path: str):
        """Save detailed results to JSON"""
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {path}")


# Convenience function for full evaluation pipeline
def run_full_evaluation(model, tokenizer, test_dataset, 
                        history: Optional[Dict] = None,
                        training_time: Optional[float] = None,
                        output_dir: str = "/kaggle/working/reports/") -> Dict[str, Any]:
    """
    Run complete evaluation pipeline and generate all outputs.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        test_dataset: Test dataset
        history: Training history dict (optional)
        training_time: Total training time in seconds (optional)
        output_dir: Where to save results
    
    Returns:
        Dictionary with all evaluation artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = TaylorEvaluator(model, tokenizer)
    
    # Run evaluation
    result = evaluator.evaluate(test_dataset)
    
    # Save results
    evaluator.save_results(result, "/kaggle/working/metrics.json")
    
    # Visualize attention (if LSTM with attention)
    evaluator.visualize_attention(test_dataset, num_samples=5)
    
    # Generate sample predictions file
    with open("/kaggle/working/FASEROH/working/sample_predictions.json", 'w') as f:
        json.dump(result.predictions[:100], f, indent=2)  # Save first 100
    
    artifacts = {
        'result': result,
        'output_dir': output_dir,
        'history': history,
        'training_time': training_time
    }
    
    return artifacts


if __name__ == "__main__":
    # Example usage demonstration
    print("Taylor Expansion Evaluator")
    print("=" * 60)
    
    # Test symbolic verifier
    verifier = SymbolicVerifier()
    
    test_cases = [
        ("1/2*x", "x/2"),
        ("x**2 + 2*x + 1", "(x+1)**2"),
        ("sin(x)", "sin ( x )"),  # Whitespace difference
        ("x - x**3/6", "x - 1/6*x**3"),
    ]
    
    print("\nSymbolic Verification Tests:")
    for pred, truth in test_cases:
        exact, symbolic = verifier.check_equivalence(pred, truth)
        print(f"  Pred: {pred:20s} | Truth: {truth:20s} | "
              f"Exact: {exact} | Symbolic: {symbolic}")
