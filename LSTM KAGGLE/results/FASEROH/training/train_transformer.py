
import logging
import argparse
from pathlib import Path
import torch
import sys
# sys.path.append("/teamspace/studios/this_studio/")

import sys
import os
# Dynamically find the project root instead of hardcoding /teamspace/
sys.path.append(os.getcwd())

# 1. Import your Phase 2 Data Pipeline
from FASEROH.data.dataset import build_complete_pipeline 

# 2. Import your Phase 3 & 4 Configs, Factory, and Trainer
from FASEROH.training.utils import ModelConfig, TrainingConfig, TaylorModelFactory, train_model

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for Transformer training."""
    parser = argparse.ArgumentParser(description="Train Transformer Model for Taylor Expansions")
    
    # Data and Path arguments
    parser.add_argument("--data_dir", type=str, default="data/raw_datasets", help="Directory containing JSONL data")
    parser.add_argument("--output_dir", type=str, default="data/tokenized_data", help="Where to save tokenized arrays")
    parser.add_argument("--checkpoint_dir", type=str, default="results/transformer_checkpoints", help="Where to save model weights")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    
    # Transformer-specific arguments
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of encoder/decoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward network dimension")
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal", choices=["sinusoidal", "learned"], help="Type of positional encoding")
    
    return parser.parse_args()

def main():
    # 1. Grab the arguments from the terminal
    args = parse_args()
    logger.info(f"Setting up Transformer Training Pipeline with args: {args}")

    # 2. Setup Data Pipeline
    data_path = Path(args.data_dir)
    pipeline, metadata = build_complete_pipeline(
        train_jsonl="/kaggle/input/datasets/iamogh/faseroh/train_data.jsonl",
        test_jsonl="/kaggle/input/datasets/iamogh/faseroh/test_data.jsonl",
        output_dir=args.output_dir
    )
    
    train_loader = pipeline.create_pytorch_dataset(split='train')
    val_loader = pipeline.create_pytorch_dataset(split='test')

    # 3. Configure and Build the Transformer
    model_config = ModelConfig(
        vocab_size=metadata['vocab_size_output'], 
        d_model=args.d_model, 
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        positional_encoding=args.pos_encoding,
        pad_idx=metadata['special_tokens']['pad']
    )
    
    # The factory fetches the Transformer architecture
    model = TaylorModelFactory.create_model("transformer", model_config)

    # 4. Configure Training Strategy
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
        checkpoint_dir=args.checkpoint_dir,
        use_curriculum=True,
        monitor_metric="loss"  # The safe metric key we fixed earlier
    )

    # # --- RESUME FROM CHECKPOINT ---
    # checkpoint_path = "results/transformer_checkpoints/checkpoint_latest.pt" # Put your exact filename here
    
    # print(f"\n🚀 Resuming training from {checkpoint_path}...")
    # checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    # # Load the model's memory
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optional but highly recommended: Load the optimizer's memory so learning rate momentum is preserved!
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    # ------------------------------

    # 5. Execute Training
    trainer = train_model(model, train_loader, val_loader, train_config)
    
    # Save the final metrics to a JSON file
    metrics_path = Path(args.checkpoint_dir).parent / "transformer_metrics.json"
    trainer.save_history(str(metrics_path))

    # --- ADD THESE 4 LINES ---
    from FASEROH.evaluation.evaluate import run_full_evaluation
    logger.info("Starting Final Phase 5 Evaluation...")
    # Pass the Tokenizer from the pipeline so it can decode the outputs!
    run_full_evaluation(model, pipeline.output_tokenizer, val_loader, output_dir="results/lstm_evaluation")
    # -------------------------

if __name__ == "__main__":
    main()
