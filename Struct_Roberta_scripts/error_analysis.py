# error_analysis.py (Version with --checkpoint_path support)
import argparse
import os
import torch
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
# Ensure these point to the correct model/data definitions for StructRoberta
# You might need to adjust these imports based on your exact file names (e.g., models_rs.py, data_rs.py)
from models_rs import StructBiaffineParser # Or models.py if that's where StructBiaffineParser is defined
from data_rs import DependencyDataset, get_collate_fn # Or data.py

def predict_heads_and_labels(model, data_loader, device):
    """Generates predictions for heads and labels."""
    model.eval()
    all_pred_heads, all_gold_heads = [], []
    all_pred_labels, all_gold_labels = [], []
    all_lengths = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_heads = batch['heads']
            gold_labels = batch['labels']

            arc_scores, label_scores = model(input_ids, attention_mask)

            pred_heads = arc_scores.argmax(dim=-1).cpu()
            pred_labels_full = label_scores.argmax(dim=-1).cpu()

            clamped_pred_heads = pred_heads.clamp(min=0)
            clamped_pred_heads_expanded = clamped_pred_heads.unsqueeze(-1)
            pred_labels = torch.gather(pred_labels_full, 2, clamped_pred_heads_expanded).squeeze(-1)

            lengths = (gold_heads != -100).sum(dim=1) - 1
            all_lengths.append(lengths.tolist())

            all_pred_heads.append(pred_heads)
            all_gold_heads.append(gold_heads)
            all_pred_labels.append(pred_labels)
            all_gold_labels.append(gold_labels)

    flat_pred_heads, flat_gold_heads = [], []
    flat_pred_labels, flat_gold_labels = [], []

    for b_ph, b_gh, b_pl, b_gl, b_len_list in zip(all_pred_heads, all_gold_heads, all_pred_labels, all_gold_labels, all_lengths):
        for i in range(b_ph.shape[0]):
            seq_len = b_len_list[i]
            if seq_len <= 0: continue
            flat_pred_heads.extend(b_ph[i, 1:seq_len+1].tolist())
            flat_gold_heads.extend(b_gh[i, 1:seq_len+1].tolist())
            flat_pred_labels.extend(b_pl[i, 1:seq_len+1].tolist())
            flat_gold_labels.extend(b_gl[i, 1:seq_len+1].tolist())

    return flat_pred_heads, flat_gold_heads, flat_pred_labels, flat_gold_labels

def main():
    parser = argparse.ArgumentParser(description="Dependency Label Error Analysis")
    # --- Arguments for .ckpt/.pt loading ---
    parser.add_argument('--model_dir', required=True, help="Directory containing label2id.json")
    parser.add_argument('--checkpoint_path', required=True, help="Path to the model .ckpt or .pt file")
    # --- Standard Arguments ---
    parser.add_argument('--data', required=True, help="Path to the test .conllu file")
    parser.add_argument('--pretrained_model', required=True, help="Path or name of the BASE pre-trained model architecture")
    parser.add_argument('--tokenizer_path', required=True, help="Path to the tokenizer used during training.") # Added tokenizer path
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_n_errors', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--conf_matrix_csv', type=str, default="confusion_matrix.csv")
    parser.add_argument('--top_errors_csv', type=str, default="top_errors.csv")
    parser.add_argument('--label_acc_csv', type=str, default="label_accuracy.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Resources ---
    print("Loading resources...")
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    add_prefix = 'roberta' in args.tokenizer_path.lower() or 'xlm' in args.tokenizer_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_prefix_space=add_prefix)

    label_map_path = os.path.join(args.model_dir, 'label2id.json')
    if not os.path.exists(label_map_path):
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        label_map_path_alt = os.path.join(ckpt_dir, 'label2id.json')
        if os.path.exists(label_map_path_alt):
             label_map_path = label_map_path_alt
             print(f"Found label2id.json in checkpoint directory: {ckpt_dir}")
        else:
             print(f"Error: label2id.json not found in {args.model_dir} or {ckpt_dir}")
             return

    with open(label_map_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {i: label for label, i in label2id.items()}
    n_labels = len(label2id)
    label_names = [id2label[i] for i in range(n_labels)]
    print(f"Loaded label map with {n_labels} labels.")

    print("Loading test data...")
    test_data = DependencyDataset(args.data, tokenizer, label2id=label2id)
    collate_fn_with_tokenizer = get_collate_fn(tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn_with_tokenizer)

    # --- Load Model (MODIFIED FOR .ckpt/.pt) ---
    print(f"Loading base model architecture from: {args.pretrained_model}")
    # Initialize the correct Parser Class (assuming StructBiaffineParser for StructRoberta)
    # The encoder is loaded *inside* the StructBiaffineParser class now
    model = StructBiaffineParser(model_name=args.pretrained_model, n_labels=n_labels).to(device)

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    print(f"Loading weights from checkpoint/PT file: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Check if the checkpoint is a state_dict directly or has a 'model' key
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict_to_load = checkpoint['model']
        print("Loading weights from 'model' key in checkpoint.")
    elif isinstance(checkpoint, dict) and 'encoder' in checkpoint: # Handle case where only encoder was saved
        state_dict_to_load = checkpoint
        print("Checkpoint seems to contain encoder weights directly. Loading with strict=False.")
        load_result = model.load_state_dict(state_dict_to_load, strict=False)
        print("Model state_dict loaded from checkpoint. Load result:", load_result)
        state_dict_to_load = None # Prevent loading again below
    elif isinstance(checkpoint, dict): # Assume it's a raw state_dict
         state_dict_to_load = checkpoint
         print("Loading weights directly from checkpoint dictionary (assumed state_dict).")
    else:
        print("Error: Unknown checkpoint format.")
        return

    if state_dict_to_load:
        # Load the weights. strict=False is safer when loading partial weights (.pt)
        # or if the checkpoint structure doesn't perfectly match.
        load_result = model.load_state_dict(state_dict_to_load, strict=False)
        print("Model state_dict loaded from checkpoint. Load result:", load_result)

    print("Model loaded successfully.")
    # --- End Model Loading Modification ---

    # --- Generate Predictions & Analysis (remains the same) ---
    pred_heads, gold_heads, pred_labels, gold_labels = predict_heads_and_labels(model, test_loader, device)
    # ... (rest of the script: confusion matrix, saving CSVs, etc.) ...
    print("Building confusion matrix (considering only correct attachments)...")
    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)
    correct_attachment_count = 0
    total_valid_tokens = 0
    pad_label_id = label2id.get('<pad>', -1)
    underscore_label_id = label2id.get('_', -1)

    for p_head, g_head, p_label, g_label in zip(pred_heads, gold_heads, pred_labels, gold_labels):
        if g_head == -100 or g_label == pad_label_id or g_label == underscore_label_id:
             continue
        total_valid_tokens += 1
        if p_head == g_head:
            correct_attachment_count += 1
            if 0 <= g_label < n_labels and 0 <= p_label < n_labels:
                 conf_matrix[g_label, p_label] += 1
            else:
                 print(f"Warning: Invalid label ID detected. Gold: {g_label}, Pred: {p_label}. Skipping.")

    calculated_uas = (correct_attachment_count / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0
    print("\n--- Confusion Matrix (Rows: Gold Labels, Columns: Predicted Labels) ---")
    print(f"Analysis based on {correct_attachment_count}/{total_valid_tokens} tokens with correctly predicted heads (UAS: {calculated_uas:.2f}%).")

    conf_df = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
    conf_matrix_path = os.path.join(args.output_dir, args.conf_matrix_csv)
    conf_df.to_csv(conf_matrix_path)
    print(f"Confusion matrix saved to: {conf_matrix_path}")

    errors = []
    for r in range(n_labels):
        for c in range(n_labels):
            if r != c and conf_matrix[r, c] > 0:
                errors.append({'Gold': id2label[r], 'Predicted': id2label[c], 'Count': conf_matrix[r, c]})
    errors_df = pd.DataFrame(errors)
    errors_df = errors_df.sort_values(by='Count', ascending=False)
    top_errors_path = os.path.join(args.output_dir, args.top_errors_csv)
    errors_df.head(args.top_n_errors).to_csv(top_errors_path, index=False)
    print(f"Top {args.top_n_errors} errors saved to: {top_errors_path}")
    print(f"\n--- Top {args.top_n_errors} Label Confusions (Gold -> Predicted | Count) ---")
    print(errors_df.head(args.top_n_errors).to_string(index=False))

    label_accuracies = []
    correct_labels_diag = np.diag(conf_matrix)
    total_gold_labels_per_row = conf_matrix.sum(axis=1)
    for i in range(n_labels):
        label_name = id2label[i]
        total_gold = total_gold_labels_per_row[i]
        correct_pred = correct_labels_diag[i]
        accuracy = (correct_pred / total_gold) * 100 if total_gold > 0 else float('nan')
        label_accuracies.append({
            'Label': label_name,
            'Accuracy (%)': f"{accuracy:.2f}" if not np.isnan(accuracy) else 'N/A',
            'Correct': correct_pred,
            'Total': total_gold
        })
    label_acc_df = pd.DataFrame(label_accuracies)
    label_acc_path = os.path.join(args.output_dir, args.label_acc_csv)
    label_acc_df.to_csv(label_acc_path, index=False)
    print(f"Per-label accuracy saved to: {label_acc_path}")
    print("\n--- Per-Label Accuracy (Based on Correct Attachments) ---")
    print(label_acc_df.to_string(index=False))

if __name__ == "__main__":
    main()