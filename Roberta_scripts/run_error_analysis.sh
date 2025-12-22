#!/bin/bash

# Ensure CUDA is visible if using GPU (optional, can be removed if using CPU)
export CUDA_VISIBLE_DEVICES=0

# Set PYTHONPATH to include the src directory so imports work
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# --- Configuration ---
# 1. Directory containing label2id.json (often the same as where checkpoints are)
#    (!!! This should point to the directory shown in your screenshot !!!)
MODEL_DIR="src/fintunedmodels/hindi_parser_model" # <<--- UPDATE THIS PATH

# 2. Path to the specific checkpoint file to load
#    (!!! Choose the checkpoint you want to evaluate, e.g., the last one !!!)
CHECKPOINT_PATH="src/fintunedmodels/hindi_parser_model/step-48700.ckpt" # <<--- UPDATE THIS PATH

# 3. Path to the TEST data file (.conllu format)
TEST_DATA="src/data/hi_hdtb-ud-test.conllu" # CHANGE AS NEEDED

# 4. Path or name of the BASE pre-trained model used for fine-tuning
#    (!!! Make sure this matches the base model used to create these checkpoints !!!)
BASE_MODEL="src/models/Roberta_hi" # <<--- UPDATE THIS PATH (e.g., could be XLM-R)

# 5. Path to the error analysis script
ERROR_SCRIPT_PATH="src/roberta_hi_scripts/error_analysis.py" # CHANGE AS NEEDED

# 6. Directory to save output CSVs
OUTPUT_CSV_DIR="src/roberta_hi_scripts/error_analysis_results_ckpt" # Changed to avoid overwriting

# 7. (Optional) Filenames for output CSVs
CONF_MATRIX_FILENAME="roberta_hi_ckpt_confusion_matrix.csv"
TOP_ERRORS_FILENAME="roberta_hi_ckpt_top_errors.csv"
LABEL_ACC_FILENAME="roberta_hi_ckpt_label_accuracy.csv"

# 8. (Optional) How many top errors to display
TOP_N_ERRORS=20

# --- Create Output Directory ---
mkdir -p $OUTPUT_CSV_DIR

# --- Run the Error Analysis Script ---
echo "Running Dependency Label Error Analysis from Checkpoint..."
python3 $ERROR_SCRIPT_PATH \
  --model_dir $MODEL_DIR \
  --checkpoint_path $CHECKPOINT_PATH \
  --data $TEST_DATA \
  --pretrained_model $BASE_MODEL \
  --top_n_errors $TOP_N_ERRORS \
  --output_dir $OUTPUT_CSV_DIR \
  --conf_matrix_csv $CONF_MATRIX_FILENAME \
  --top_errors_csv $TOP_ERRORS_FILENAME \
  --label_acc_csv $LABEL_ACC_FILENAME

echo "Error analysis complete. Results saved in $OUTPUT_CSV_DIR"