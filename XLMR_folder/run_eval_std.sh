#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/src:$PYTHONPATH


MODEL_DIR="src/fintunedmodels/XLMR_Pfinetuned"

python3 src/XLMR_scripts/eval_std.py \
  --model_dir $MODEL_DIR \
  --data src/test_NonLVC_auto.conllu \
  --pretrained_model src/models/xlm-roberta-base-local \
  --tokenizer_path src/hindi_RobertaTok \
