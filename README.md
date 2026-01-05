# LLaMA Bengali Empathetic Conversations Fine-Tuning

Fine-tune Meta LLaMA 3.1 8B Instruct on the Bengali Empathetic Conversations Corpus using parameter-efficient LoRA adapters and 4-bit loading via [Unsloth](https://github.com/unslothai/unsloth). The pipeline covers data prep, training, evaluation, and inference for Bengali conversational assistants.

## Features
- 4-bit loading with LoRA adapters (r=16) for memory-efficient finetuning
- Automatic dataset formatting into the model chat template with lightweight filtering
- Training with TRL `SFTTrainer` and cosine LR schedule
- Evaluation scripts for ROUGE, BLEU, and perplexity plus quick text generation CLI
- Centralized configuration in `config.py` for model paths, hyperparameters, and data locations

## Repository Structure
- `config.py` — model, LoRA, data, and training configuration knobs
- `data_processor.py` — loads the Bengali Empathetic Conversations CSV, formats prompts/responses, and creates train/val/test splits
- `model_loader.py` — loads the base LLaMA model/tokenizer and applies LoRA adapters
- `trainer.py` — builds the TRL SFT trainer with logging and evaluation callbacks
- `train.py` — end-to-end training entrypoint (load data/model, train, save final adapter)
- `evaluate.py` — computes ROUGE/BLEU/perplexity on held-out samples from the dataset
- `generate.py` — CLI to generate responses from the fine-tuned model
- `experiments/` — notebooks for experimentation
- `models/` — checkpoints and final saved model
- `outputs/` — place for logs/metrics (if you choose to write them)

## Requirements
- Python 3.10+ and a CUDA-enabled GPU (4-bit + LoRA keeps memory needs modest, but a 24GB GPU is recommended)
- `pip install -r requirements.txt`
- The Bengali Empathetic Conversations CSV at `data/BengaliEmpatheticConversationsCorpus.csv` (default path, configurable)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation
- Expected CSV columns: question (required), answer (required), optional topic/title columns. Column names are detected case-insensitively.
- Rows with missing questions/answers or very short answers are skipped.
- Data is formatted into a chat-style prompt/response (`tokenizer.apply_chat_template`) and split deterministically into train/validation/test subsets.
- Adjust the dataset path in `config.DATA_PATH` if your CSV lives elsewhere.

## Training
```bash
cd llama-bengali-finetune
python train.py
```
- Loads `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (configurable via `config.MODEL_NAME`).
- Applies LoRA with `r=16`, alpha=16, dropout=0.05 on attention and MLP projections.
- Key hyperparameters (edit in `config.py`): `NUM_EPOCHS=0.5`, `BATCH_SIZE=4`, `GRADIENT_ACCUMULATION_STEPS=2`, `LEARNING_RATE=2e-4`, `EVAL_STEPS=200`, `MAX_SEQ_LENGTH=1024`.
- Checkpoints write to `models/checkpoints/`; the final adapter and tokenizer save to `models/final/`.

## Evaluation
```bash
python evaluate.py
```
- Loads the fine-tuned model from `config.FINAL_MODEL_DIR`.
- Generates predictions for `config.NUM_TEST_SAMPLES` examples and reports ROUGE-1/2/L, BLEU, and perplexity (computed on up to `config.MAX_PERPLEXITY_SAMPLES` samples).

## Inference
```bash
python generate.py --prompt "বাংলায় আপনার প্রশ্ন লিখুন..." --max-tokens 256
```
- Loads the saved model from `models/final/` and returns a sampled response. Adjust decoding parameters inside `generate.py` if needed.

## Configuration Tips
- Update `config.py` to point to alternate base models, change LoRA target modules, or tune training/evaluation parameters.
- Ensure `config.CHECKPOINT_DIR`, `config.FINAL_MODEL_DIR`, and `config.OUTPUT_DIR` exist or are creatable before running long jobs.
- Set `LOAD_IN_4BIT=False` if you prefer full-precision loading (requires more GPU memory).

## Notes
- Deterministic seeds are set where supported, but minor metric variance can occur across runs.
- If you plan to share artifacts, include licensing/usage details for the dataset and the base model according to their respective terms.
