import argparse, os, torch, json, sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, XLMRobertaModel
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- 1. Correct Imports for this model ---
from models_std import BiaffineParser
from data_std import DependencyDataset, get_collate_fn

# --- 2. CORE ANALYSIS FUNCTION (Identical to the one above) ---
def analyze_performance(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Analyzing Performance"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_heads = batch['heads']
            gold_labels = batch['labels']

            arc_scores, label_scores = model(input_ids, attention_mask)
            pred_heads = arc_scores.argmax(dim=-1)

            clamped_pred_heads = pred_heads.clamp(min=0)
            batch_size, seq_len, _, n_labels = label_scores.shape
            clamped_pred_heads_for_gather = clamped_pred_heads.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, n_labels)
            pred_labels_at_pred_heads_scores = torch.gather(label_scores, 2, clamped_pred_heads_for_gather).squeeze(2)
            pred_labels = pred_labels_at_pred_heads_scores.argmax(dim=-1)

            gold_heads = gold_heads.cpu().numpy()
            gold_labels = gold_labels.cpu().numpy()
            pred_heads = pred_heads.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()

            for i in range(input_ids.shape[0]):
                mask = (gold_heads[i] != -100) & (gold_heads[i] != 0)
                sent_len = int(mask.sum())
                if sent_len == 0: continue
                g_heads = gold_heads[i][mask]
                g_labels = gold_labels[i][mask]
                p_heads = pred_heads[i][mask]
                p_labels = pred_labels[i][mask]
                token_indices = np.where(mask)[0] 

                for j in range(sent_len):
                    token_idx = token_indices[j]
                    gold_head_idx = g_heads[j]
                    if gold_head_idx < 0: continue
                    uas_correct = (gold_head_idx == p_heads[j])
                    las_correct = uas_correct and (g_labels[j] == p_labels[j])
                    distance = abs(token_idx - gold_head_idx)
                    direction = 'right' if token_idx < gold_head_idx else 'left'
                    if distance == 0: direction = 'self' 
                    results.append({
                        'uas': uas_correct, 'las': las_correct,
                        'distance': distance, 'direction': direction, 'sent_len': sent_len
                    })
    return pd.DataFrame(results)
# --- END OF CORE ANALYSIS FUNCTION ---

def main():
    parser = argparse.ArgumentParser(description="Baseline XLMR Distance/Direction Analysis")
    parser.add_argument('--model_dir', default='src/fintunedmodels/XLMR_finetuned')
    parser.add_argument('--test_data', required=True, help="Path to the test .conllu file")
    parser.add_argument('--pretrained_model', default='src/models/xlm-roberta-base-local')
    parser.add_argument('--tokenizer_path', default='src/hindi_RobertaTok')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', default='distance_error_analysis_XLMR_baseline')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_prefix_space=True)

    label_map_path = os.path.join(args.model_dir, 'label2id.json')
    with open(label_map_path, 'r') as f: label2id = json.load(f)
    n_labels = len(label2id)

    test_data = DependencyDataset(args.test_data, tokenizer, label2id=label2id)
    collate_fn = get_collate_fn(tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Loading Baseline BiaffineParser for {args.model_dir}...")
    # --- 3. Correct Model Loading for this model ---
    encoder = XLMRobertaModel.from_pretrained(args.pretrained_model).to(device)
    model = BiaffineParser(encoder, n_labels=n_labels).to(device)
    # ---
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pt'), map_location=device))
    print("Model loaded successfully.")

    # --- Run Analysis ---
    df = analyze_performance(model, test_loader, device)
    
    if df.empty:
        print("Analysis produced no results. Check data and model.")
        sys.exit(1)

    # --- 4. CORE ANALYSIS BLOCK (Identical to the one above) ---
    print("\n--- Analyzing by Distance ---")
    dist_bins = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20, max(30, df['distance'].max())]
    df['dist_bin'] = pd.cut(df['distance'], bins=dist_bins, right=True)
    dist_analysis = (df.groupby('dist_bin')[['uas', 'las']].mean() * 100).round(2)
    dist_analysis['count'] = df.groupby('dist_bin')['uas'].count()
    
    print("--- Analyzing by Direction ---")
    dir_analysis = (df.groupby('direction')[['uas', 'las']].mean() * 100).round(2)
    dir_analysis['count'] = df.groupby('direction')['uas'].count()

    print("--- Analyzing by Sentence Length ---")
    len_bins = [0, 10, 20, 30, 40, 50, max(60, df['sent_len'].max())]
    df['len_bin'] = pd.cut(df['sent_len'], bins=len_bins, right=True)
    len_analysis = (df.groupby('len_bin')[['uas', 'las']].mean() * 100).round(2)
    len_analysis['count'] = df.groupby('len_bin')['uas'].count()
    # --- END OF CORE ANALYSIS BLOCK ---

    # --- Save Results ---
    model_name = "baseline_xlmr"
    dist_path = os.path.join(args.output_dir, f'{model_name}_distance_analysis.csv')
    dir_path = os.path.join(args.output_dir, f'{model_name}_direction_analysis.csv')
    len_path = os.path.join(args.output_dir, f'{model_name}_length_analysis.csv')

    dist_analysis.to_csv(dist_path)
    dir_analysis.to_csv(dir_path)
    len_analysis.to_csv(len_path)

    print(f"\n✅ {model_name} Distance/Direction Analysis Complete")
    print(f"Results saved to {args.output_dir}")
    print("\nDistance Analysis:\n", dist_analysis)
    print("\nDirection Analysis:\n", dir_analysis)
    print("\nLength Analysis:\n", len_analysis)

if __name__ == "__main__":
    main()