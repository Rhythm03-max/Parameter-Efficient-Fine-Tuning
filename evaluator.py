"""Evaluation utilities"""

import torch
import numpy as np
import evaluate


class Evaluator:
    """Compute evaluation metrics"""
    
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('sacrebleu')
    
    def compute_metrics(self, predictions, references):
        """Calculate ROUGE and BLEU"""
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        bleu_scores = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        return {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bleu': bleu_scores['score'] / 100.0,
        }
    
    def compute_perplexity(self, model, tokenizer, dataset, max_samples=100):
        """Calculate perplexity"""
        model.eval()
        total_loss = 0
        count = 0
        
        print(f"Computing perplexity on {min(max_samples, len(dataset))} samples...")
        
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                
                text = example.get('text', '')
                if not text:
                    continue
                
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                try:
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    total_loss += outputs.loss.item()
                    count += 1
                except:
                    continue
        
        if count == 0:
            return float('inf')
        
        avg_loss = total_loss / count
        perplexity = np.exp(avg_loss)
        
        print(f"Perplexity: {perplexity:.4f}\n")
        return perplexity