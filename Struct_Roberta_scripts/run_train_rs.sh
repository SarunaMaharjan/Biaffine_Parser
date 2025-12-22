#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Define paths
SAVE_DIR="src/fintunedmodels/roberta_hi_mod_Pparser"
CUSTOM_MODEL_PATH="src/models/struct_roberta_final.pt" # IMPORTANT: Set this path
BASE_MODEL_NAME="src/models/Roberta_hi" # Or the specific roberta model you used
TOKENIZER_PATH="src/hindi_RobertaTok"  # Or the path to your Hindi tokenizer

python3 src/struct_roberta_hi_scripts/main_rs.py \
  --train src/data/hi_hdtb-ud-train.conllu \
  --dev src/data/hi_hdtb-ud-dev.conllu \
  --custom_pretrained_model $CUSTOM_MODEL_PATH \
  --base_model_name $BASE_MODEL_NAME \
  --tokenizer_path $TOKENIZER_PATH \
  --epochs 20 \
  --batch_size 8 \
  --lr 5e-5 \
  --save_dir $SAVE_DIR