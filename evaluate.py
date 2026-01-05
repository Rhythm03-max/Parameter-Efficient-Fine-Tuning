"""Evaluation script"""

import torch
from unsloth import FastLanguageModel
from data_processor import DatasetProcessor
from evaluator import Evaluator
import config


def main():
    print("="*60)
    print("Model Evaluation")
    print("="*60 + "\n")
    
    # Load fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.FINAL_MODEL_DIR,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load test data
    processor = DatasetProcessor(config.DATA_PATH, tokenizer)
    _, _, test_ds = processor.load_and_preprocess()
    
    # Generate predictions
    FastLanguageModel.for_inference(model)
    
    predictions = []
    references = []
    
    print(f"Generating {config.NUM_TEST_SAMPLES} predictions...\n")
    
    for i, example in enumerate(test_ds.select(range(config.NUM_TEST_SAMPLES))):
        user_content = example.get('user_content', '')
        reference = example.get('answer', '')
        
        if not user_content or not reference:
            continue
        
        messages = [{"role": "user", "content": user_content}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(reference)
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{config.NUM_TEST_SAMPLES}...")
    
    # Compute metrics
    evaluator = Evaluator()
    
    metrics = evaluator.compute_metrics(predictions, references)
    perplexity = evaluator.compute_perplexity(
        model, tokenizer, test_ds, 
        max_samples=config.MAX_PERPLEXITY_SAMPLES
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"ROUGE-1:    {metrics['rouge1']:.4f}")
    print(f"ROUGE-2:    {metrics['rouge2']:.4f}")
    print(f"ROUGE-L:    {metrics['rougeL']:.4f}")
    print(f"BLEU:       {metrics['bleu']:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()