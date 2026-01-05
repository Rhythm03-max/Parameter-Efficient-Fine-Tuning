"""Configuration settings"""

# Model Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Configuration
NUM_EPOCHS = 0.5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
EVAL_STEPS = 200

# Paths
DATA_PATH = "./data/BengaliEmpatheticConversationsCorpus.csv"
CHECKPOINT_DIR = "./models/checkpoints"
FINAL_MODEL_DIR = "./models/final"
OUTPUT_DIR = "./outputs"

# Evaluation
NUM_TEST_SAMPLES = 50
MAX_PERPLEXITY_SAMPLES = 100