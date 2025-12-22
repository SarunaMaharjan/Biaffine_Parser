#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Use a clear name for the save directory
SAVE_DIR="src/fintunedmodels/XLMR_Pfinetuned"

python3 src/XLMR_scripts/main_std.py \
  --train src/data/hi_hdtb-ud-train.conllu \
  --dev src/data/hi_hdtb-ud-dev.conllu \
  --pretrained_model src/models/xlm-roberta-base-local \
  --tokenizer_path src/hindi_RobertaTok \
  --epochs 10 \
  --batch_size 4 \
  --lr 5e-5 \
  --save_dir $SAVE_DIR