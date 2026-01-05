"""Data processing utilities"""

import pandas as pd
import numpy as np
from datasets import Dataset


class DatasetProcessor:
    """Process Bengali Empathetic Conversations dataset"""
    
    def __init__(self, dataset_path, tokenizer):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.stats = {}
    
    def load_and_preprocess(self):
        """Load CSV and create formatted dataset"""
        print("\n" + "="*60)
        print("Loading Dataset")
        print("="*60 + "\n")
        
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}\n")
        
        # Map columns
        col_map = {col.lower().strip(): col for col in df.columns}
        
        topic_col = None
        title_col = None
        question_col = None
        answer_col = None
        
        for key, col in col_map.items():
            if 'topic' in key:
                topic_col = col
            elif 'title' in key:
                title_col = col
            elif 'question' in key and 'title' not in key:
                question_col = col
            elif 'answer' in key:
                answer_col = col
        
        if not question_col or not answer_col:
            raise ValueError(f"Missing required columns! Found: {df.columns.tolist()}")
        
        # Format data
        formatted_data = []
        skipped = 0
        lengths = []
        
        for idx, row in df.iterrows():
            topic = str(row[topic_col]).strip() if topic_col and pd.notna(row[topic_col]) else ""
            title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
            question = str(row[question_col]).strip() if pd.notna(row[question_col]) else ""
            answer = str(row[answer_col]).strip() if pd.notna(row[answer_col]) else ""
            
            if not question or not answer or len(answer) < 10:
                skipped += 1
                continue
            
            # Build context
            context_parts = []
            if topic and topic != 'nan':
                context_parts.append(f"Topic: {topic}")
            if title and title != 'nan':
                context_parts.append(f"Title: {title}")
            context_parts.append(f"Question: {question}")
            
            user_content = "\n".join(context_parts)
            
            # Apply chat template
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer}
                    ],
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                formatted_data.append({
                    "text": formatted_text,
                    "user_content": user_content,
                    "answer": answer
                })
                
                lengths.append(len(formatted_text))
                
            except Exception as e:
                print(f"Error formatting row {idx}: {e}")
                skipped += 1
                continue
        
        print(f"Formatted: {len(formatted_data)} examples")
        print(f"Skipped: {skipped} examples")
        print(f"Avg text length: {np.mean(lengths):.0f} chars\n")
        
        # Create splits
        dataset = Dataset.from_list(formatted_data)
        subset_60 = dataset.train_test_split(test_size=0.6, seed=42)['train']
        train_test = subset_60.train_test_split(test_size=0.1, seed=42)
        val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
        
        self.stats = {
            'total': len(formatted_data),
            'train': len(train_test['train']),
            'val': len(val_test['train']),
            'test': len(val_test['test'])
        }
        
        print(f"Train: {self.stats['train']}")
        print(f"Val: {self.stats['val']}")
        print(f"Test: {self.stats['test']}\n")
        
        return train_test['train'], val_test['train'], val_test['test']