#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

MODEL_DIR="src/fintunedmodels/roberta_hi_mod_Pparser"  # Path to your fine-tuned model
BASE_MODEL_NAME="src/models/Roberta_hi" # Or the specific roberta model you used
TOKENIZER_PATH="src/hindi_RobertaTok"  # Or the path to your Hindi tokenizer

echo "--- Running Evaluation on Test Set ---"
python3 src/struct_roberta_hi_scripts/test_rs.py \
  --model_dir $MODEL_DIR \
  --data src/test_NonLVC_auto.conllu \
  --base_model_name $BASE_MODEL_NAME \
  --tokenizer_path $TOKENIZER_PATH