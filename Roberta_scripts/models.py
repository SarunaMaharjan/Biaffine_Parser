# src/models.py
import torch
import torch.nn as nn

class Biaffine(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.U = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        self.W1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W2 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b)

    def forward(self, x1, x2):
        bilinear_term = torch.einsum('bid,odl,bjl->boij', x1, self.U, x2)
        linear_dep_term = torch.einsum('bid,od->boi', x1, self.W1)
        linear_head_term = torch.einsum('bjd,od->boj', x2, self.W2)
        scores = bilinear_term + linear_dep_term.unsqueeze(3) + linear_head_term.unsqueeze(2) + self.b.view(1, -1, 1, 1)
        return scores.permute(0, 2, 3, 1).squeeze(-1)

class BiaffineParser(nn.Module):
    def __init__(self, encoder, arc_mlp_size=500, label_mlp_size=100, n_labels=50):
        super().__init__()
        self.encoder = encoder
        
        # Get hidden_size dynamically from the encoder's config
        hidden_size = self.encoder.config.hidden_size
        
        # MLPs for arc prediction
        self.arc_mlp_head = nn.Linear(hidden_size, arc_mlp_size)
        self.arc_mlp_dep = nn.Linear(hidden_size, arc_mlp_size)
        
        # MLPs for label prediction
        self.label_mlp_head = nn.Linear(hidden_size, label_mlp_size)
        self.label_mlp_dep = nn.Linear(hidden_size, label_mlp_size)
        
        # Biaffine classifiers
        self.arc_biaffine = Biaffine(arc_mlp_size, 1)
        self.label_biaffine = Biaffine(label_mlp_size, n_labels)
        
        self.dropout = nn.Dropout(0.33)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        # Pass inputs through the encoder (e.g., roberta_hi)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(outputs.last_hidden_state)

        # Apply MLPs
        arc_h = self.relu(self.arc_mlp_head(h))
        arc_d = self.relu(self.arc_mlp_dep(h))
        label_h = self.relu(self.label_mlp_head(h))
        label_d = self.relu(self.label_mlp_dep(h))
        
        arc_h, arc_d = self.dropout(arc_h), self.dropout(arc_d)
        label_h, label_d = self.dropout(label_h), self.dropout(label_d)

        # Get scores
        arc_scores = self.arc_biaffine(arc_d, arc_h)
        label_scores = self.label_biaffine(label_d, label_h)
        
        return arc_scores, label_scores