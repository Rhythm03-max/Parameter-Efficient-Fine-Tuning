"""Training utilities"""

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported
import config


class MetricsCallback(TrainerCallback):
    """Track training metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append({
                    'step': state.global_step,
                    'loss': logs['loss']
                })
            if 'eval_loss' in logs:
                self.eval_losses.append({
                    'step': state.global_step,
                    'loss': logs['eval_loss']
                })


def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    """Setup SFTTrainer"""
    
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=config.WARMUP_RATIO,
        learning_rate=config.LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        dataloader_num_workers=2,
    )
    
    metrics_callback = MetricsCallback()
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,
        args=training_args,
        callbacks=[metrics_callback],
    )
    
    print("\nTrainer configured successfully!\n")
    return trainer, metrics_callback