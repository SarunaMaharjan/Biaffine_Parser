#!/usr/bin/env bash
set -e

# --- 1. Environment Setup ---
# Set GPU visibility (adjust device index if needed)
export CUDA_VISIBLE_DEVICES=0
# Ensure Python can find local modules (src/struct_xlmr.py, etc.)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# --- 2. Configuration Variables ---

# CHECKPOINT: Path to your pre-trained StructXLMR model weights
# IMPORTANT: This must be the path to your struct_xlmr.pt file
CUSTOM_MODEL_PATH=src/models/struct_xlmr_final.pt 

# BASE_MODEL_NAME: Path/name of the base RoBERTa/XLMR model configuration
# Used by the tokenizer and StructBiaffineParser to build the architecture
BASE_MODEL_NAME=src/models/xlm-roberta-base-local

# TOKENIZER_PATH: Path to your custom Hindi tokenizer
TOKENIZER_PATH=src/hindi_RobertaTok

# OUT_DIR: Where the fine-tuned models and label map (label2id.json) will be saved
SAVE_DIR=src/fintunedmodels/Struct_XLMR_Parser_PFinetuned 

# DATA_DIR: Base directory for the CoNLL-U files
DATA_DIR=src/data 

# --- 3. Execution Command ---

echo "Starting StructXLMR Dependency Parsing Fine-Tuning..."
echo "Output directory: ${SAVE_DIR}"
echo "Pre-trained checkpoint: ${CUSTOM_MODEL_PATH}"
echo "---"

# Call the main script (assuming main_parser.py is inside src/)
python src/Struct_xlmr_scripts/main_parser.py \
  --train ${DATA_DIR}/hi_hdtb-ud-train.conllu \
  --dev ${DATA_DIR}/hi_hdtb-ud-dev.conllu \
  --custom_pretrained_model ${CUSTOM_MODEL_PATH} \
  --base_model_name ${BASE_MODEL_NAME} \
  --tokenizer_path ${TOKENIZER_PATH} \
  --epochs 20 \
  --batch_size 16 \
  --lr 3e-5 \
  --save_dir ${SAVE_DIR} 
  # Note: The original argument structure used only one --lr. 
  # If you want dual LR, you need to modify the argparse and the shell script accordingly.

echo "---"
echo "Training process completed. Best model saved to ${SAVE_DIR}"