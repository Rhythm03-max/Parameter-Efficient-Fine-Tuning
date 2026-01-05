"""Main training script"""

import torch
from data_processor import DatasetProcessor
from model_loader import load_model, apply_lora
from trainer import setup_trainer
import config


def main():
    print("="*60)
    print("LLaMA Bengali Fine-Tuning")
    print("="*60)
    
    # Check GPU
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, tokenizer = load_model()
    
    # Load data
    processor = DatasetProcessor(config.DATA_PATH, tokenizer)
    train_ds, val_ds, test_ds = processor.load_and_preprocess()
    
    # Apply LoRA
    model = apply_lora(model)
    
    # Setup trainer
    trainer, metrics_callback = setup_trainer(model, tokenizer, train_ds, val_ds)
    
    # Train
    print("\n" + "="*60)
    print("Training Started")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\nTraining Complete!")
    
    # Save model
    print(f"\nSaving model to {config.FINAL_MODEL_DIR}...")
    model.save_pretrained(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    print("Model saved!\n")
    
    print("="*60)
    print("Training Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()