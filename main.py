# src/main.py
import argparse, os, torch, json, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from models import BiaffineParser
from data import DependencyDataset, get_collate_fn
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss, pbar = 0, tqdm(data_loader, desc="Training")
    for batch in pbar:
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        gold_heads, gold_labels = batch['heads'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        arc_scores, label_scores = model(input_ids, attention_mask)
        arc_loss_fct, label_loss_fct = nn.CrossEntropyLoss(ignore_index=-100), nn.CrossEntropyLoss(ignore_index=-100)
        arc_loss = arc_loss_fct(arc_scores.view(-1, arc_scores.size(-1)), gold_heads.view(-1))
        clamped_gold_heads = gold_heads.clamp(min=0)
        batch_size, seq_len, n_labels = label_scores.shape[0], label_scores.shape[1], label_scores.shape[-1]
        clamped_g_h_gather = clamped_gold_heads.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, n_labels)
        label_scores_at_gold_heads = torch.gather(label_scores, 2, clamped_g_h_gather).squeeze(2)
        label_loss = label_loss_fct(label_scores_at_gold_heads.view(-1, n_labels), gold_labels.view(-1))
        loss = arc_loss + label_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_uas, total_las, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            gold_heads, gold_labels = batch['heads'], batch['labels']
            arc_scores, label_scores = model(input_ids, attention_mask)
            pred_heads = arc_scores.argmax(dim=-1).cpu()
            pred_labels = label_scores.argmax(dim=-1).cpu()
            clamped_pred_heads = pred_heads.clamp(min=0)
            pred_labels_at_pred_heads = torch.gather(pred_labels, 2, clamped_pred_heads.unsqueeze(-1)).squeeze(-1)
            mask = (gold_heads != -100) & (gold_heads != 0)
            uas_correct = (pred_heads[mask] == gold_heads[mask]).sum().item()
            las_correct = ((pred_heads[mask] == gold_heads[mask]) & (pred_labels_at_pred_heads[mask] == gold_labels[mask])).sum().item()
            total_uas, total_las, total_tokens = total_uas + uas_correct, total_las + las_correct, total_tokens + mask.sum().item()
    uas = (total_uas / total_tokens) * 100 if total_tokens > 0 else 0
    las = (total_las / total_tokens) * 100 if total_tokens > 0 else 0
    return uas, las

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--pretrained_model', required=True, help="Path to your roberta_hi model or name on Hub")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, add_prefix_space=True)
    train_data = DependencyDataset(args.train, tokenizer)

    label_map_path = os.path.join(args.save_dir, 'label2id.json')
    with open(label_map_path, 'w') as f: json.dump(train_data.label2id, f, indent=4)
    print(f"Saved label map with {train_data.num_labels} labels to {label_map_path}")

    dev_data = DependencyDataset(args.dev, tokenizer, label2id=train_data.label2id)
    
    collate_fn_with_tokenizer = get_collate_fn(tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_tokenizer)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_fn_with_tokenizer)

    encoder = AutoModel.from_pretrained(args.pretrained_model)
    model = BiaffineParser(encoder, n_labels=train_data.num_labels).to(device)

    for param in model.encoder.parameters():
       param.requires_grad = False
    print("!!! ENCODER FROZEN: Performing Probing (Before Fine-Tuning) !!!")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
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