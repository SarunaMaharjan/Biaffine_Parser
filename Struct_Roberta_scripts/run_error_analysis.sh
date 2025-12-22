#!/bin/bash

# Ensure CUDA is visible if using GPU (optional, can be removed if using CPU)
export CUDA_VISIBLE_DEVICES=0

# Set PYTHONPATH to include the src directory so imports work
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# --- Configuration for StructRoberta (.pt) ---
# 1. Directory containing label2id.json
#    (!!! This should be where main_rs.py saved it, e.g., roberta_hi_mod_parser !!!)
LABEL_MAP_DIR="src/fintunedmodels/roberta_hi_mod_parser" # <<--- UPDATE THIS PATH

# 2. Path to the SPECIFIC .pt weight file to load
CHECKPOINT_PATH="src/models/struct_roberta_final.pt" # <<--- VERIFY THIS PATH

# 3. Path to the TEST data file (.conllu format)
TEST_DATA="src/data/hi_hdtb-ud-test.conllu"

# 4. Path or name of the BASE pre-trained model architecture
BASE_MODEL="src/models/Roberta_hi" # <<--- VERIFY THIS PATH

# 5. Define the Tokenizer Path
#    (!!! Should match the tokenizer used when fine-tuning StructRoberta !!!)
TOKENIZER_PATH="src/hindi_RobertaTok" # <<--- UPDATE THIS PATH (Check run_train_rs.sh)

# 6. Path to the error analysis script
ERROR_SCRIPT_PATH="src/struct_roberta_hi_scripts/error_analysis.py" # <<--- UPDATE THIS PATH if needed

# 7. Directory to save output CSVs
OUTPUT_CSV_DIR="src/struct_roberta_hi_scripts/error_analysis_results_pt"

# 8. Filenames for output CSVs
CONF_MATRIX_FILENAME="struct_roberta_pt_confusion_matrix.csv"
TOP_ERRORS_FILENAME="struct_roberta_pt_top_errors.csv"
LABEL_ACC_FILENAME="struct_roberta_pt_label_accuracy.csv"

# 9. How many top errors to display
TOP_N_ERRORS=20

# --- Create Output Directory ---
mkdir -p $OUTPUT_CSV_DIR

# --- Run the Error Analysis Script ---
echo "Running Dependency Label Error Analysis for StructRoberta (.pt)..."
python3 $ERROR_SCRIPT_PATH \
  --model_dir $LABEL_MAP_DIR \
  --checkpoint_path $CHECKPOINT_PATH \
  --data $TEST_DATA \
  --pretrained_model $BASE_MODEL \
  --tokenizer_path $TOKENIZER_PATH \
  --top_n_errors $TOP_N_ERRORS \
  --output_dir $OUTPUT_CSV_DIR \
  --conf_matrix_csv $CONF_MATRIX_FILENAME \
  --top_errors_csv $TOP_ERRORS_FILENAME \
  --label_acc_csv $LABEL_ACC_FILENAME

echo "-------Results Saved-----------"