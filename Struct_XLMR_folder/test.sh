#!/usr/bin/env bash
set -e

# --- 1. Environment Setup ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH # Assumes your scripts are in src/

# --- 2. Configuration Variables ---

# Directory where training output (best_model.pt and label2id.json) was saved.
MODEL_DIR=src/fintunedmodels/Struct_XLMR_Parser_PFinetuned

# Base model name for architecture config
BASE_MODEL_NAME=src/models/xlm-roberta-base-local

# Path to your custom Hindi tokenizer
TOKENIZER_PATH=src/hindi_RobertaTok

# Base directory for the CoNLL-U files
DATA_DIR=src

# The file to evaluate on
TEST_FILE=${DATA_DIR}/test_NonLVC_auto.conllu

# --- 3. Execution Command ---

echo "Starting Simple (Argmax) Evaluation..."
echo "Model Dir: ${MODEL_DIR}"
echo "Test File: ${TEST_FILE}"
echo "---"

python src/Struct_xlmr_scripts/eval_parser.py \
  --model_dir ${MODEL_DIR} \
  --data ${TEST_FILE} \
  --base_model_name ${BASE_MODEL_NAME} \
  --tokenizer_path ${TOKENIZER_PATH} \
  --batch_size 16

echo "---"
echo "Evaluation complete."
