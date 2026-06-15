<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:001f3f,50:0074D9,100:7FDBFF&height=180&section=header&text=Parameter-Efficient%20Fine-Tuning&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=LLM-Driven%20Telephony%20Agent%20%7C%20Twilio%20%C2%B7%20ElevenLabs%20%C2%B7%20OpenAI&descAlignY=58&descSize=18&descColor=f4a8c9
" width="100%"/>



<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-4--bit%20LoRA-7c3aed?style=flat-square)
![TRL](https://img.shields.io/badge/TRL-SFTTrainer-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-00C7B7?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

**A parameter-efficient fine-tuning pipeline that adapts Meta LLaMA 3.1 8B Instruct into a Bengali empathetic conversational assistant — using 4-bit quantization and LoRA adapters via Unsloth, with built-in ROUGE/BLEU/perplexity evaluation and a CLI for inference.**

</div>

---

## Overview

This project fine-tunes **`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`** on the **Bengali Empathetic Conversations Corpus** — a dataset of Bengali-language emotional support questions and empathetic responses — to produce a lightweight conversational assistant capable of responding supportively in Bengali.

The entire lifecycle is covered: **dataset loading & formatting → 4-bit model loading → LoRA adapter injection → SFT training with TRL → evaluation (ROUGE/BLEU/perplexity) → CLI inference**. Every stage is a small, independently runnable script driven by a single centralized `config.py`, making it straightforward to swap the base model, adjust LoRA targets, or retune training hyperparameters without touching pipeline code.

---

## Pipeline Architecture

```
        data/raw/BengaliEmpatheticConversationsCorpus.csv
                          │
                          ▼
              ┌─────────────────────────┐
              │   DatasetProcessor        │  ── Column detection (Topics/Title/
              │   (data_processor.py)     │     Questions/Answers), filtering,
              │                           │     chat-template formatting
              └────────────┬──────────────┘
                           │
              ┌────────────┴──────────────┐
              ▼                            ▼
        train / val / test            (deterministic seed=42 splits)
              │
              ▼
              ┌─────────────────────────┐
              │     load_model()          │  ── FastLanguageModel.from_pretrained
              │   (model_loader.py)        │     4-bit NF4 quantized LLaMA 3.1 8B
              └────────────┬──────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │      apply_lora()          │  ── LoRA r=16, alpha=16, dropout=0.05
              │   (model_loader.py)         │     on attention + MLP projections
              └────────────┬──────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │     setup_trainer()        │  ── TRL SFTTrainer, cosine LR,
              │     (trainer.py)            │     bf16/fp16, packing, eval/save steps
              └────────────┬──────────────┘
                           │
                           ▼
                    trainer.train()
                           │
                           ▼
              models/final/  (LoRA adapter + tokenizer)
                           │
              ┌────────────┴──────────────┐
              ▼                            ▼
      evaluate.py                    generate.py
  (ROUGE / BLEU / perplexity)     (CLI inference / chat)
```

---

## Project Structure

```text
LLama_fine_Tune/
├── config.py            # Centralized configuration — model, LoRA, training, paths
├── data_processor.py     # DatasetProcessor — load CSV, format chat template, split
├── model_loader.py        # load_model() + apply_lora() — 4-bit base model & LoRA
├── trainer.py              # setup_trainer() — TRL SFTTrainer + MetricsCallback
├── train.py                 # End-to-end training entrypoint
├── evaluator.py              # Evaluator — ROUGE, BLEU (sacrebleu), perplexity
├── evaluate.py                # Evaluation entrypoint — runs Evaluator on test split
├── generate.py                 # CLI for single-prompt inference with the fine-tuned model
├── experiments/
│   └── LLaMA_3_1_8B_Instruct.ipynb   # Exploratory / experimentation notebook
├── data/
│   └── raw/
│       └── BengaliEmpatheticConversationsCorpus.csv   # Source dataset (38k+ rows)
├── requirements.txt          # Python dependencies (torch, transformers, trl, unsloth, ...)
└── README.md
```

> `models/` (checkpoints + final adapter) and `outputs/` (logs/metrics) are created at runtime by the training and evaluation scripts and are not checked into the repository.

---

## Component Details

### `config.py` — Centralized Configuration

Every tunable knob lives here:

| Category | Settings |
|---|---|
| **Model** | `MODEL_NAME`, `MAX_SEQ_LENGTH=1024`, `LOAD_IN_4BIT=True` |
| **LoRA** | `LORA_R=16`, `LORA_ALPHA=16`, `LORA_DROPOUT=0.05`, `LORA_TARGET_MODULES` (q/k/v/o + gate/up/down projections) |
| **Training** | `NUM_EPOCHS=0.5`, `BATCH_SIZE=4`, `GRADIENT_ACCUMULATION_STEPS=2`, `LEARNING_RATE=2e-4`, `WARMUP_RATIO=0.03`, `EVAL_STEPS=200` |
| **Paths** | `DATA_PATH`, `CHECKPOINT_DIR`, `FINAL_MODEL_DIR`, `OUTPUT_DIR` |
| **Evaluation** | `NUM_TEST_SAMPLES=50`, `MAX_PERPLEXITY_SAMPLES=100` |

### `data_processor.py` — `DatasetProcessor`

- **Case-insensitive column detection** — automatically locates `Topics`, `Question-Title`/`Title`, `Questions`, and `Answers` columns regardless of exact naming, so the dataset CSV can vary slightly in schema
- **Filtering** — drops rows with missing questions/answers or answers shorter than 10 characters
- **Prompt construction** — builds a user turn from available context (`Topic: ...`, `Title: ...`, `Question: ...`) paired with the answer as the assistant turn, formatted via `tokenizer.apply_chat_template`
- **Deterministic splitting** — uses a fixed `seed=42` two-stage split: first takes a 40% subset of the full dataset, then splits that into train/val/test

### `model_loader.py`

- **`load_model()`** — loads the base model and tokenizer via `FastLanguageModel.from_pretrained` with 4-bit NF4 quantization and `max_seq_length=1024`
- **`apply_lora()`** — injects LoRA adapters via `FastLanguageModel.get_peft_model`, targeting all attention and MLP projection layers, with `use_gradient_checkpointing="unsloth"` for memory efficiency, and prints total/trainable parameter counts and the trainable percentage

### `trainer.py`

- **`setup_trainer()`** — configures a TRL `SFTTrainer` with `TrainingArguments`: cosine LR schedule, `adamw_8bit` optimizer, automatic bf16/fp16 selection via `is_bfloat16_supported()`, sequence packing, gradient checkpointing, `group_by_length=True`, and best-model checkpointing on `eval_loss`
- **`MetricsCallback`** — a `TrainerCallback` that records per-step training and evaluation losses for later inspection

### `evaluator.py` — `Evaluator`

- **`compute_metrics()`** — computes ROUGE-1/2/L (via `evaluate`'s `rouge` metric with stemming) and BLEU (via `sacrebleu`, normalized to a 0–1 scale)
- **`compute_perplexity()`** — runs the model in eval mode over up to `MAX_PERPLEXITY_SAMPLES` examples, computing average loss and exponentiating to perplexity

### `train.py` / `evaluate.py` / `generate.py`

Thin orchestration scripts that wire the components above into runnable end-to-end pipelines for training, evaluation, and single-prompt inference respectively.

---

## Dataset

**Bengali Empathetic Conversations Corpus** (`data/raw/BengaliEmpatheticConversationsCorpus.csv`)

| Column | Description |
|---|---|
| `Topics` | Broad emotional/topical category (e.g. পারিবারিক দ্বন্দ্ব — "family conflict") |
| `Question-Title` | Short title summarizing the question |
| `Questions` | The user's question or situation, in Bengali |
| `Answers` | An empathetic, supportive response in Bengali |

The corpus contains **38,000+ rows**. After filtering, the data is formatted into chat-style `{"role": "user", ...} / {"role": "assistant", ...}` pairs and split deterministically (seed `42`) using a 40% subset → 90/5/5 train/val/test partition.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-enabled GPU — 4-bit + LoRA keeps memory needs modest, but **24GB VRAM is recommended**
- The dataset CSV at `data/raw/BengaliEmpatheticConversationsCorpus.csv` (path configurable via `config.DATA_PATH`)

### Installation

```bash
git clone https://github.com/Rhythm05Roy/LLama_fine_Tune.git
cd LLama_fine_Tune

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` installs Unsloth directly from GitHub (`unsloth @ git+https://github.com/unslothai/unsloth.git`), along with PyTorch, Transformers, TRL, PEFT, bitsandbytes, and evaluation libraries (`evaluate`, `rouge-score`, `sacrebleu`, `nltk`).

---

## Usage

### 1. Training

```bash
python train.py
```

This will:
1. Check CUDA availability and print the detected GPU
2. Load `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` in 4-bit (configurable via `config.MODEL_NAME`)
3. Load and format the dataset via `DatasetProcessor`
4. Apply LoRA adapters (`r=16`, `alpha=16`, `dropout=0.05` on attention + MLP projections)
5. Train with TRL `SFTTrainer` using a cosine LR schedule
6. Save checkpoints to `models/checkpoints/` and the final adapter + tokenizer to `models/final/`

### 2. Evaluation

```bash
python evaluate.py
```

- Loads the fine-tuned model from `config.FINAL_MODEL_DIR`
- Generates predictions for `config.NUM_TEST_SAMPLES` held-out examples
- Reports **ROUGE-1**, **ROUGE-2**, **ROUGE-L**, **BLEU**, and **perplexity** (computed on up to `config.MAX_PERPLEXITY_SAMPLES` samples)

### 3. Inference

```bash
python generate.py --prompt "বাংলায় আপনার প্রশ্ন লিখুন..." --max-tokens 256
```

Loads the saved model from `models/final/` and returns a sampled response (`temperature=0.7`, `top_p=0.9`). Adjust decoding parameters directly in `generate.py` if needed.

---

## Configuration Tips

- **Different base model** — change `config.MODEL_NAME` to any Unsloth-compatible 4-bit model
- **LoRA targets** — edit `LORA_TARGET_MODULES` to fine-tune a different subset of layers (e.g. attention-only for lighter adapters)
- **Full precision** — set `LOAD_IN_4BIT = False` if you have sufficient VRAM and prefer full-precision loading
- **Output directories** — ensure `CHECKPOINT_DIR`, `FINAL_MODEL_DIR`, and `OUTPUT_DIR` exist or are creatable before launching long training jobs
- **Training length** — `NUM_EPOCHS=0.5` is intentionally short for quick experimentation; increase for a fuller fine-tune

---

## Experiments

`experiments/LLaMA_3_1_8B_Instruct.ipynb` contains the exploratory notebook used during development — useful as a reference for interactive experimentation with the same model/data before running the scripted pipeline.

---

## Notes

- Deterministic seeds (`42`) are set where supported, but minor metric variance can still occur across runs and hardware
- If sharing trained artifacts, include licensing/usage details for both the **Bengali Empathetic Conversations Corpus** and the **LLaMA 3.1** base model according to their respective terms

---

## License

MIT — see `LICENSE`.

<div align="center">

**Built by [Ridam Roy](https://github.com/Rhythm05Roy)**

[![GitHub](https://img.shields.io/badge/GitHub-Rhythm05Roy-181717?style=flat-square&logo=github)](https://github.com/Rhythm05Roy)
[![Email](https://img.shields.io/badge/Email-ridam15--4260%40diu.edu.bd-D44638?style=flat-square&logo=gmail&logoColor=white)](mailto:ridam15-4260@diu.edu.bd)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:001f3f,50:0074D9,100:7FDBFF&height=80&section=footer" width="100%"/>

</div>
