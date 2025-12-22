import argparse, os, torch, json, sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- 1. Correct Imports for this model ---
from struct_xlmr import StructBiaffineParser
from data_parser import DependencyDataset, get_collate_fn

def analyze_performance(model, data_loader, device):
    """
    Runs the model and collects UAS, LAS, Distance, Direction, and Sent Length
    for every token in the dataset.
    """
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

            # 4D Label Score Logic [B, D, H, L]
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
                # --- START: FIX ---
                # The new mask *includes* root tokens (where gold_head == 0)
                # It only filters out padding/ignored tokens (-100)
                mask = (gold_heads[i] != -100)
                # --- END: FIX ---
                
                sent_len = int(mask.sum())
                if sent_len == 0:
                    continue

                g_heads = gold_heads[i][mask]
                g_labels = gold_labels[i][mask]
                p_heads = pred_heads[i][mask]
                p_labels = pred_labels[i][mask]
                
                # Get the indices of the actual tokens
                token_indices = np.where(mask)[0] 

                for j in range(sent_len):
                    token_idx = token_indices[j]
                    gold_head_idx = g_heads[j]
                    
                    # We only care about valid, in-sentence heads
                    if gold_head_idx < 0: # Should not happen with our mask, but as a safeguard
                        continue
                        
                    uas_correct = (gold_head_idx == p_heads[j])
                    las_correct = uas_correct and (g_labels[j] == p_labels[j])
                    
                    # --- START: FIX ---
                    # Explicitly check for root tokens to categorize them as 'self'
                    if gold_head_idx == 0:
                        distance = 0
                        direction = 'self'
                    else:
                        # Calculate distance and direction for all non-root tokens
                        distance = abs(token_idx - gold_head_idx)
                        direction = 'right' if token_idx < gold_head_idx else 'left'
                    # --- END: FIX ---

                    results.append({
                        'uas': uas_correct,
                        'las': las_correct,
                        'distance': distance,
                        'direction': direction,
                        'sent_len': sent_len
                    })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="StructXLMR Distance/Direction Analysis")
    parser.add_argument('--model_dir', default='src/fintunedmodels/Struct_XLMR_Parser_Finetuned')
    parser.add_argument('--test_data', required=True, help="Path to the test .conllu file")
    parser.add_argument('--pretrained_model', default='src/models/xlm-roberta-base-local')
    parser.add_argument('--tokenizer_path', default='src/hindi_RobertaTok')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', default='src/Struct_xlmr_scripts/Struct_XLMR_distance_error_analysis')
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

    print(f"Loading StructBiaffineParser for {args.model_dir}...")
    model = StructBiaffineParser(model_name=args.pretrained_model, n_labels=n_labels).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pt'), map_location=device))
    print("Model loaded successfully.")

    # --- Run Analysis ---
    df = analyze_performance(model, test_loader, device)
    
    if df.empty:
        print("Analysis produced no results. Check data and model.")
        sys.exit(1)

    # --- 1. Distance Analysis ---
    print("\n--- Analyzing by Distance ---")
    # --- START: FIX ---
    # Bin 0 (which is now 'self') must be handled separately.
    # We create bins for 1+ distance.
    dist_bins = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20, max(30, df[df['distance'] > 0]['distance'].max())]
    # Create the 'dist_bin' column only for non-self (distance > 0) tokens
    df_non_self = df[df['distance'] > 0].copy()
    df_non_self['dist_bin'] = pd.cut(df_non_self['distance'], bins=dist_bins, right=True)
    
    # Group and analyze non-root tokens
    dist_analysis = (df_non_self.groupby('dist_bin', observed=True)[['uas', 'las']].mean() * 100).round(2)
    dist_analysis['count'] = df_non_self.groupby('dist_bin', observed=True)['uas'].count()
    # --- END: FIX ---
    
    # --- 2. Direction Analysis ---
    print("--- Analyzing by Direction ---")
    # This will now include 'self' automatically
    dir_analysis = (df.groupby('direction')[['uas', 'las']].mean() * 100).round(2)
    dir_analysis['count'] = df.groupby('direction')['uas'].count()

    # --- 3. Sentence Length Analysis ---
    print("--- Analyzing by Sentence Length ---")
    len_bins = [0, 10, 20, 30, 40, 50, max(60, df['sent_len'].max())]
    df['len_bin'] = pd.cut(df['sent_len'], bins=len_bins, right=True)
    len_analysis = (df.groupby('len_bin', observed=True)[['uas', 'las']].mean() * 100).round(2)
    len_analysis['count'] = df.groupby('len_bin', observed=True)['uas'].count()

    # --- Save Results ---
    model_name = "struct_xlmr"
    dist_path = os.path.join(args.output_dir, f'{model_name}_distance_analysis.csv')
    dir_path = os.path.join(args.output_dir, f'{model_name}_direction_analysis.csv')
    len_path = os.path.join(args.output_dir, f'{model_name}_length_analysis.csv')

    dist_analysis.to_csv(dist_path)
    dir_analysis.to_csv(dir_path)
    len_analysis.to_csv(len_path)

    print(f"\n✅ {model_name} Distance/Direction Analysis Complete")
    print(f"Results saved to {args.output_dir}")
    print("\nDistance Analysis (Excluding root):\n", dist_analysis)
    print("\nDirection Analysis (Including root as 'self'):\n", dir_analysis)
    print("\nLength Analysis:\n", len_analysis)

if __name__ == "__main__":
    main()
