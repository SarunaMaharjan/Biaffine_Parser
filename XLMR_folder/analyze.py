import argparse
import os
import torch
import json
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, AutoTokenizer # Use XLMRobertaModel for consistency
# --- START: MODEL/DATA FIX ---
# Import from your "std" (standard/baseline) files
from models_std import BiaffineParser
from data_std import DependencyDataset, get_collate_fn
# --- END: MODEL/DATA FIX ---
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper function for error analysis ---
def analyze_errors(model, data_loader, device, id2label, debug_samples=0):
    model.eval()
    label_error_counts = defaultdict(lambda: defaultdict(int))
    label_error_breakdown = defaultdict(lambda: {'Head Errors': 0, 'Label-only Errors': 0})
    total_tokens = 0

    uas_total = 0
    uas_correct = 0
    las_total = 0
    las_correct = 0

    debug_printed = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Analyzing Errors"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_heads = batch['heads']
            gold_labels = batch['labels']

            arc_scores, label_scores = model(input_ids, attention_mask)
            pred_heads = arc_scores.argmax(dim=-1)

            # --- START: LOGIC FIX (Reverted to 4D-aware logic) ---
            # The BiaffineParser returns 4D label scores: [B, D, H, L]
            # We must select the label based on the *predicted head* (pred_heads).
            
            # Label prediction is conditioned on the predicted head
            clamped_pred_heads = pred_heads.clamp(min=0)
            
            # Use torch.gather to select label scores at the predicted head indices
            batch_size, seq_len, _, n_labels = label_scores.shape # Unpack 4D shape
            
            clamped_pred_heads_for_gather = clamped_pred_heads.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, n_labels)
            pred_labels_at_pred_heads_scores = torch.gather(label_scores, 2, clamped_pred_heads_for_gather).squeeze(2) # Shape [B, D, L]
            
            # Now we get the final predicted label ID
            pred_labels = pred_labels_at_pred_heads_scores.argmax(dim=-1) # Shape [B, D]
            # --- END: LOGIC FIX ---

            gold_heads = gold_heads.cpu()
            gold_labels = gold_labels.cpu()
            pred_heads = pred_heads.cpu()
            pred_labels = pred_labels.cpu() # This is now correct

            for i in range(input_ids.shape[0]):
                # Mask out padding (-100) and ROOT (0) tokens
                mask = (gold_heads[i] != -100)
                g_heads = gold_heads[i][mask].tolist()
                g_labels = gold_labels[i][mask].tolist()
                p_heads = pred_heads[i][mask].tolist()
                p_labels = pred_labels[i][mask].tolist()

                for j in range(len(g_labels)):
                    total_tokens += 1
                    g_label = id2label.get(g_labels[j], "OOB_GOLD")
                    p_label = id2label.get(p_labels[j], "OOB_PRED")
                    head_correct = g_heads[j] == p_heads[j]
                    label_correct = g_label == p_label

                    # 1. Confusion Matrix Count
                    label_error_counts[g_label][p_label] += 1

                    # 2. LAS/UAS Aggregation
                    uas_total += 1
                    las_total += 1
                    if head_correct:
                        uas_correct += 1
                    if head_correct and label_correct:
                        las_correct += 1
                    
                    # 3. Detailed Error Breakdown (per Gold Label)
                    if not head_correct:
                        # Head Error (UAS Error)
                        label_error_breakdown[g_label]['Head Errors'] += 1
                    elif not label_correct:
                        # Label-only Error: Head is correct, but label is wrong
                        label_error_breakdown[g_label]['Label-only Errors'] += 1

            # Debug print
            if debug_printed < debug_samples:
                print(f"\nSample {debug_printed+1} (Masked Tokens)")
                print(f"Gold heads: {g_heads}")
                print(f"Pred heads: {p_heads}")
                print(f"Gold labels: {[id2label.get(x, 'OOB') for x in g_labels]}")
                print(f"Pred labels: {[id2label.get(x, 'OOB') for x in p_labels]}")
                debug_printed += 1

    uas_pct = (uas_correct / uas_total) * 100 if uas_total > 0 else 0.0
    las_pct = (las_correct / las_total) * 100 if las_total > 0 else 0.0

    print("\n========== SUMMARY ==========")
    print(f"Total tokens analyzed: {total_tokens}")
    print(f"UAS (Head correct only)   : {uas_correct} / {uas_total} -> {uas_pct:.2f}%")
    print(f"LAS (Head + Label correct): {las_correct} / {las_total} -> {las_pct:.2f}%")
    print("=============================\n")

    return label_error_counts, label_error_breakdown, uas_pct, las_pct

# --- Visualization (functions are fine) ---
def plot_confusion_matrix(confusion_df, output_path):
    plt.figure(figsize=(20, 18))
    df_to_plot = confusion_df.copy()
    np.fill_diagonal(df_to_plot.values, np.nan) 
    sns.heatmap(df_to_plot, annot=False, cmap="viridis", fmt=".1f", linewidths=.5, cbar=True)
    plt.title('Dependency Label Confusion Matrix (Gold vs. Predicted) - Errors Only')
    plt.ylabel('Gold Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_confusions(confusion_df, top_n, output_path):
    confusion_pairs = []
    for gold in confusion_df.index:
        for pred in confusion_df.columns:
            if gold != pred and confusion_df.loc[gold, pred] > 0:
                confusion_pairs.append((f'{gold} -> {pred}', confusion_df.loc[gold, pred]))
    confusion_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = confusion_pairs[:top_n]
    if not top_pairs:
        return
    pairs, counts = zip(*top_pairs)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=list(pairs), palette="Reds_d")
    plt.title(f'Top {top_n} Dependency Label Misclassifications')
    plt.xlabel('Error Count')
    plt.ylabel('Confusion Pair (Gold -> Predicted)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="XLM-R Dependency Label Error Analysis")
    parser.add_argument('--model_dir', default='src/fintunedmodels/XLMR_finetuned',
                        help='Directory with fine-tuned model weights and label2id.json')
    parser.add_argument('--pretrained_model', default='src/models/xlm-roberta-base-local',
                        help='Base XLM-R pretrained model (HuggingFace name or local path)')
    parser.add_argument('--tokenizer_path', default='src/hindi_RobertaTok',
                        help='Path to the saved tokenizer')
    parser.add_argument('--test_data', default='src/data/hi_hdtb-ud-test.conllu', help='Path to test CONLLU file')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', default='error_analysis_XLMR_baseline1') # Changed output dir
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_prefix_space=True)
    print(f"Loaded tokenizer from: {args.tokenizer_path}")

    # Load labels
    label_map_path = os.path.join(args.model_dir, 'label2id.json')
    with open(label_map_path, 'r') as f:
        label2id = json.load(f)
    # --- START: ID2LABEL FIX ---
    # Create the map from ID -> Label
    id2label = {int(v): k for k, v in label2id.items()}
    # --- END: ID2LABEL FIX ---
    n_labels = len(label2id)
    print(f"Loaded {n_labels} labels.")

    # Load test data
    test_data = DependencyDataset(args.test_data, tokenizer, label2id=label2id)
    collate_fn = get_collate_fn(tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # Load encoder and model
    print(f"Loading encoder from: {args.pretrained_model}")
    encoder = XLMRobertaModel.from_pretrained(args.pretrained_model)
    print(f"Initializing BiaffineParser...")
    model = BiaffineParser(encoder, n_labels=n_labels).to(device)

    # Load fine-tuned weights
    model_path = os.path.join(args.model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded fine-tuned model weights from {model_path}")

    # Run analysis
    label_error_counts, label_error_breakdown, uas_pct, las_pct = analyze_errors(model, test_loader, device, id2label)

    # Confusion matrix
    gold_labels = sorted([l for l in id2label.values() if l not in ['<pad>', '_', 'OOB_GOLD', 'OOB_PRED']])
    confusion_matrix = pd.DataFrame(0, index=gold_labels, columns=gold_labels)
    for g_label, preds in label_error_counts.items():
        for p_label, count in preds.items():
            if g_label in confusion_matrix.index and p_label in confusion_matrix.columns:
                confusion_matrix.loc[g_label, p_label] = count

    # Calculate final label metrics
    label_metrics = []
    for label in gold_labels:
        total = confusion_matrix.loc[label].sum()
        correct = confusion_matrix.loc[label, label]
        errors = total - correct
        acc = (correct / total * 100) if total > 0 else 0
        
        head_errors = label_error_breakdown[label]['Head Errors']
        label_only_errors = label_error_breakdown[label]['Label-only Errors']
        
        top_conf = confusion_matrix.loc[label].drop(label, errors='ignore').sort_values(ascending=False).head(3)
        top_conf_str = '; '.join([f"{pred}: {cnt}" for pred, cnt in top_conf.items() if cnt > 0])
        
        label_metrics.append({
            'Label': label,
            'Total Instances': int(total),
            'Correct': int(correct),
            'Head Errors (UAS Fail)': int(head_errors), 
            'Label-only Errors (LAS Fail @ Correct Head)': int(label_only_errors),
            'Error Count (Total LAS Fail)': int(errors),
            'Accuracy (%)': acc,
            'Top Confusions (Pred: Count)': top_conf_str
        })

    metrics_df = pd.DataFrame(label_metrics).sort_values(by='Label-only Errors (LAS Fail @ Correct Head)', ascending=False)
    
    # Save results
    metrics_df.to_csv(os.path.join(args.output_dir, 'xlmr_baseline_hi_label_metrics.csv'), index=False)
    confusion_matrix.to_csv(os.path.join(args.output_dir, 'xlmr_baseline_hi_confusion_matrix.csv'))

    # Visualizations
    error_conf_matrix = confusion_matrix.copy()
    np.fill_diagonal(error_conf_matrix.values, 0)
    confusion_percent = error_conf_matrix.div(error_conf_matrix.sum(axis=1), axis=0) * 100 
    confusion_percent = confusion_percent.fillna(0) # Handle divide-by-zero
    plot_confusion_matrix(confusion_percent, os.path.join(args.output_dir, 'xlmr_baseline_hi_confusion_heatmap.png'))
    plot_top_confusions(error_conf_matrix, 20, os.path.join(args.output_dir, 'xlmr_baseline_hi_top_confusions_bar.png'))

    print("\n✅ XLM-R Baseline Error Analysis Complete")
    print(f"Internal UAS: {uas_pct:.2f}% | Internal LAS: {las_pct:.2f}%")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

