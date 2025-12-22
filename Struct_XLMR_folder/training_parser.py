import torch
import torch.nn.functional as F
import os

class Trainer:
    def __init__(self, model, train_loader, dev_loader, device, lr_low, lr_high, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.optimizer = self._build_optimizer(lr_low, lr_high)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.best_las = 0.0

    def _build_optimizer(self, lr_low, lr_high):
        backbone_params = list(self.model.backbone.parameters())
        new_params = [p for n, p in self.model.named_parameters() if "backbone" not in n]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_low},
            {"params": new_params, "lr": lr_high}
        ], weight_decay=1e-2)
        return optimizer

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                input_ids, attention_mask, heads, rels = [b.to(self.device) for b in batch]
                arc_head_scores, arc_dep_scores, label_scores = self.model(input_ids, attention_mask)
                loss = self.compute_loss(arc_head_scores, arc_dep_scores, label_scores, heads, rels, attention_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
            uas, las = self.evaluate(epoch)
            if las > self.best_las:
                self.best_las = las
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pt"))

    def compute_loss(self, arc_head_scores, arc_dep_scores, label_scores, heads, rels, attention_mask):
        mask = attention_mask.bool()
        batch_size, seq_len, _ = arc_head_scores.size()
        arc_logits = torch.bmm(arc_dep_scores, arc_head_scores.transpose(1, 2))
        arc_loss = F.cross_entropy(arc_logits[mask], heads[mask])
        label_loss = F.cross_entropy(label_scores[mask], rels[mask])
        return arc_loss + label_loss

    def evaluate(self, epoch):
        self.model.eval()
        correct_uas = 0
        correct_las = 0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.dev_loader:
                input_ids, attention_mask, gold_heads, gold_rels = [b.to(self.device) for b in batch]
                arc_head_scores, arc_dep_scores, label_scores = self.model(input_ids, attention_mask)
                arc_logits = torch.bmm(arc_dep_scores, arc_head_scores.transpose(1, 2))
                pred_heads = arc_logits.argmax(dim=-1)
                pred_rels = label_scores.argmax(dim=-1)
                mask = attention_mask.bool()
                masked_gold_heads = gold_heads[mask]
                masked_pred_heads = pred_heads[mask]
                masked_gold_rels = gold_rels[mask]
                masked_pred_rels = pred_rels[mask]
                correct_uas += (masked_pred_heads == masked_gold_heads).sum().item()
                correct_las += ((masked_pred_heads == masked_gold_heads) & (masked_pred_rels == masked_gold_rels)).sum().item()
                total_tokens += mask.sum().item()
        uas = correct_uas / total_tokens * 100
        las = correct_las / total_tokens * 100
        print(f"Epoch {epoch} - Dev UAS: {uas:.2f}%, LAS: {las:.2f}%")
        return uas, las
