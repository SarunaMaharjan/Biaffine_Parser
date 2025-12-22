# main_parser.py (REPAIRED)
import argparse, os, torch, json, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from tqdm import tqdm

from struct_xlmr import StructBiaffineParser # This is the main model class
from data_parser import DependencyDataset, get_collate_fn 

# --- Evaluation and Training Step Functions ---

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    
    # --- START: Tweak 1 - Move loss fns out of loop ---
    arc_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    label_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    # --- END: Tweak 1 ---

    total_loss, pbar = 0, tqdm(data_loader, desc="Training")
    for batch in pbar:
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        gold_heads, gold_labels = batch['heads'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        arc_scores, label_scores = model(input_ids, attention_mask)
        
        # --- START: Safety Check for ARC_LOSS ---
        n_classes_arc = arc_scores.size(-1) # Get the sequence length
        
        # Sanity check
        if torch.any((gold_heads >= n_classes_arc) & (gold_heads != -100)):
            print(f"⚠ Invalid head indices found before clamp: {gold_heads.max().item()} >= n_classes {n_classes_arc}")
            # --- START: Tweak 3 - Add debug save ---
            torch.save({'arc_scores': arc_scores.detach().cpu(), 'label_scores': label_scores.detach().cpu(),
                        'gold_heads': gold_heads.detach().cpu(), 'gold_labels': gold_labels.detach().cpu()},
                       'bad_batch_debug_arc.pt')
            # --- END: Tweak 3 ---

        # Clamp gold_heads to be in range [0, n_classes_arc-1] or -100
        gold_heads = gold_heads.long()
        gold_heads = torch.where((gold_heads < 0) | (gold_heads >= n_classes_arc),
                                 torch.tensor(-100, device=gold_heads.device),
                                 gold_heads)
        # --- END: Safety Check for ARC_LOSS ---

        # --- Loss Calculation (Arc) ---
        arc_loss = arc_loss_fct(arc_scores.view(-1, arc_scores.size(-1)), gold_heads.view(-1))
        
        # --- Prepare for Label Loss ---
        # --- START: Tweak 2 - Replace clamp(min=0) ---
        clamped_gold_heads = gold_heads.clone()
        clamped_gold_heads[clamped_gold_heads == -100] = 0
        # --- END: Tweak 2 ---
        
        batch_size, seq_len, _, n_labels = label_scores.shape # n_labels is 27
        
        label_scores_at_gold_heads = torch.gather(
            label_scores, 
            dim=2, 
            index=clamped_gold_heads.unsqueeze(2).unsqueeze(-1).expand(batch_size, seq_len, 1, n_labels)
        ).squeeze(2) # [B, D, L]

        # --- START: Safety Check for LABEL_LOSS ---
        n_classes_label = n_labels
        
        # Sanity check
        if torch.any((gold_labels >= n_classes_label) & (gold_labels != -100)):
            print(f"⚠ Invalid label indices found before clamp: {gold_labels.max().item()} >= n_labels {n_classes_label}")
            # --- START: Tweak 3 - Add debug save ---
            torch.save({'arc_scores': arc_scores.detach().cpu(), 'label_scores': label_scores.detach().cpu(),
                        'gold_heads': gold_heads.detach().cpu(), 'gold_labels': gold_labels.detach().cpu()},
                       'bad_batch_debug_label.pt')
            # --- END: Tweak 3 ---
            
        # Clamp gold_labels to be in range [0, n_labels-1] or -100
        gold_labels = gold_labels.long()
        gold_labels = torch.where((gold_labels < 0) | (gold_labels >= n_classes_label),
                                  torch.tensor(-100, device=gold_labels.device),
                                  gold_labels)
        # --- END: Safety Check for LABEL_LOSS ---

        # --- Loss Calculation (Label) ---
        label_loss = label_loss_fct(label_scores_at_gold_heads.view(-1, n_labels), gold_labels.view(-1))
        
        # --- Total Loss ---
        loss = arc_loss + label_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """
    Evaluates UAS/LAS using simple argmax decoding (no MST/Eisner tree enforcement).
    """
    model.eval()
    total_uas, total_las, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            gold_heads, gold_labels = batch['heads'], batch['labels']
            
            arc_scores, label_scores = model(input_ids, attention_mask)
            
            # 1. Arc Prediction (Simple argmax decoding)
            pred_heads = arc_scores.argmax(dim=-1).cpu() # [B, D]

            # 2. Label Prediction: Find the best label *based on the predicted head*
            # --- START: Tweak 2 - Replace clamp(min=0) ---
            clamped_pred_heads = pred_heads.clone()
            clamped_pred_heads[clamped_pred_heads < 0] = 0 # Clamp any negatives (though argmax shouldn't produce them)
            # --- END: Tweak 2 ---
            
            pred_labels_at_pred_heads = torch.gather(
                 label_scores.argmax(dim=-1).cpu(), # [B, D, H] (Best label ID for each arc)
                 dim=2, 
                 index=clamped_pred_heads.unsqueeze(-1) # [B, D, 1]
            ).squeeze(-1) # [B, D]

            # 3. Mask and Scoring
            mask = (gold_heads != -100) & (gold_heads != 0) 
            
            uas_correct = (pred_heads[mask] == gold_heads[mask]).sum().item()
            las_correct = ((pred_heads[mask] == gold_heads[mask]) & (pred_labels_at_pred_heads[mask] == gold_labels[mask])).sum().item()
            
            total_uas += uas_correct
            total_las += las_correct
            total_tokens += mask.sum().item()
            
    uas = (total_uas / total_tokens) * 100 if total_tokens > 0 else 0
    las = (total_las / total_tokens) * 100 if total_tokens > 0 else 0
    return uas, las

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    # Arguments from run_train_rs.sh
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--custom_pretrained_model', required=True, help="Path to your struct_roberta_hi.pt file")
    parser.add_argument('--base_model_name', default='src/models/xlm-roberta-base-local', help="Base model name for architecture config")
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5) # Single LR for simplicity
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_prefix_space=True)
    
    # 1. Load Data & Build Label Map
    train_data = DependencyDataset(args.train, tokenizer)
    label_map_path = os.path.join(args.save_dir, 'label2id.json')
    with open(label_map_path, 'w') as f: json.dump(train_data.label2id, f, indent=4)
    print(f"Saved label map with {train_data.num_labels} labels to {label_map_path}")

    dev_data = DependencyDataset(args.dev, tokenizer, label2id=train_data.label2id)
    
    collate_fn_with_tokenizer = get_collate_fn(tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_tokenizer)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_fn_with_tokenizer)

    # 2. Load Model
    model = StructBiaffineParser(model_name=args.base_model_name, n_labels=train_data.num_labels).to(device)
    
    
    for param in model.encoder.parameters():
       param.requires_grad = False
    print("!!! ENCODER FROZEN: Performing Probing (Before Fine-Tuning) !!!")
# ------------------------------------
    # Load custom pretrained weights into the encoder part of the model
    pretrained_dict = torch.load(args.custom_pretrained_model, map_location=device)
    # The key to linking the StructRoberta encoder to the checkpoint is here:
    model.encoder.load_state_dict(pretrained_dict, strict=False) 
    print(f"Successfully loaded custom pretrained weights from {args.custom_pretrained_model} into the encoder.")


    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    best_las = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch} - Average Training Loss: {avg_loss:.4f}")
        
        uas, las = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch} Dev - UAS: {uas:.2f}% LAS: {las:.2f}%")

        if las > best_las:
            best_las = las
            model_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"New best LAS! Model saved to {model_path}")

if __name__ == "__main__":
    main()