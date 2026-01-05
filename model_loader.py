"""Model loading utilities"""

from unsloth import FastLanguageModel
import config


def load_model():
    """Load LLaMA model and tokenizer"""
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60 + "\n")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    
    print("Model loaded successfully!\n")
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA configuration"""
    print("\n" + "="*60)
    print("Applying LoRA")
    print("="*60 + "\n")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=config.LORA_TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )
    
    # Print parameter info
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%\n")
    
    return model