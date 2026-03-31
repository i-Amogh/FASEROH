
import logging
import argparse
from pathlib import Path
import sys
import os
# sys.path.append("/teamspace/studios/this_studio/")
sys.path.append(os.getcwd())

from FASEROH.data.dataset import build_complete_pipeline 
from FASEROH.training.utils import ModelConfig, TrainingConfig, TaylorModelFactory, train_model

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train LSTM Model for Taylor Expansions")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/raw_datasets", help="Directory containing JSONL data")
    parser.add_argument("--output_dir", type=str, default="data/tokenized_data", help="Where to save tokenized arrays")
    parser.add_argument("--checkpoint_dir", type=str, default="results/lstm_checkpoints", help="Where to save model weights")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--lstm_hidden", type=int, default=256, help="LSTM hidden state dimension")
    
    return parser.parse_args()

def main():
    # 1. Grab the arguments from the terminal
    args = parse_args()
    logger.info(f"Setting up LSTM Training Pipeline with args: {args}")

    # 2. Use the args for paths
    data_path = Path(args.data_dir)
    pipeline, metadata = build_complete_pipeline(
        # train_jsonl=str(data_path / "train_data.jsonl"), # Assuming this is what Phase 1 names it
        # test_jsonl=str(data_path / "test_data.jsonl"),
        train_jsonl="/kaggle/input/datasets/iamogh/faseroh/train_data.jsonl",
        test_jsonl="/kaggle/input/datasets/iamogh/faseroh/test_data.jsonl",
        output_dir=args.output_dir
    )
    
    train_loader = pipeline.create_pytorch_dataset(split='train')
    val_loader = pipeline.create_pytorch_dataset(split='test')

    # 3. Use the args for Model Config
    model_config = ModelConfig(
        vocab_size=metadata['vocab_size_output'], 
        d_model=args.d_model, 
        lstm_hidden=args.lstm_hidden,
        pad_idx=metadata['special_tokens']['pad']
    )
    model = TaylorModelFactory.create_model("lstm", model_config)

    # 4. Use the args for Training Config
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        use_curriculum=True,
        monitor_metric="sequence_accuracy"
    )

    # 5. Execute
    trainer = train_model(model, train_loader, val_loader, train_config)
    
    # Save metrics cleanly
    metrics_path = Path(args.checkpoint_dir).parent / "lstm_metrics.json"
    trainer.save_history(str(metrics_path))

    # --- ADD THESE 4 LINES ---
    from FASEROH.evaluation.evaluate import run_full_evaluation
    logger.info("Starting Final Phase 5 Evaluation...")
    # Pass the Tokenizer from the pipeline so it can decode the outputs!
    run_full_evaluation(model, pipeline.output_tokenizer, val_loader, output_dir="results/lstm_evaluation")
    # -------------------------

if __name__ == "__main__":
    main()
