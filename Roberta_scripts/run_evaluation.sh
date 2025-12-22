#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

MODEL_DIR="src/fintunedmodels/roberta_hi_Pfinetuned"
BASE_MODEL_NAME="src/models/Roberta_hi" # Or the specific XLM-R model you used-hi
TOKENIZER_PATH="src/hindi_RobertaTok"  # Or the path to your tokenizer

echo "--- Running Evaluation on Test Set ---"
python3 src/roberta_hi_scripts/eval.py \
  --model_dir $MODEL_DIR \
  --data src/test_NonLVC_auto.conllu \
  --pretrained_model $BASE_MODEL_NAME \
  --tokenizer_path $TOKENIZER_PATH  