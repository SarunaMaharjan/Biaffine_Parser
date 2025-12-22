#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# IMPORTANT: Set this to the path of your roberta_hi model
# This can be a local path or a name on the Hugging Face Hub
PRETRAINED_MODEL_PATH="src/models/Roberta_hi"
SAVE_DIR="src/roberta_hi_scripts/roberta_hi_Pfinetuned"

python3 src/roberta_hi_scripts/main.py \
  --train src/data/hi_hdtb-ud-train.conllu \
  --dev src/data/hi_hdtb-ud-dev.conllu \
  --pretrained_model $PRETRAINED_MODEL_PATH \
  --save_dir $SAVE_DIR