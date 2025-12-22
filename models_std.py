# src/models.py

import torch
import torch.nn as nn

class Biaffine(nn.Module):
    """
    Biaffine attention layer.
    Calculates a score for each pair of vectors in the sequence.
    """
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # U is the core bilinear weight tensor
        self.U = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        # W1 and W2 are for the linear terms (head and dependent biases)
        self.W1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W2 = nn.Parameter(torch.Tensor(out_features, in_features))
        # b is the final bias
        self.b = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights for better training stability
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b)

    def forward(self, x1, x2):
        # x1: (batch, seq_len, dim) - Dependent representation
        # x2: (batch, seq_len, dim) - Head representation

        # ** FIX APPLIED HERE **
        # The correct einsum for the bilinear term is 'bid,odl,bjd->boij'
        # to ensure the dimensions of x1 (d) and x2 (l) correctly align with U (d, l).
        # x1 ('bid'): batch, seq_dep, dim
        # U ('odl'):  out,   dim,    dim
        # x2 ('bjd'): batch, seq_head, dim
        # result ('boij'): batch, out, seq_dep, seq_head
        bilinear_term = torch.einsum('bid,odl,bjd->boij', x1, self.U, x2) # <- FIXED
        
        # Linear terms
        linear_dep_term = torch.einsum('bid,od->boi', x1, self.W1)
        linear_head_term = torch.einsum('bjd,od->boj', x2, self.W2)

        # Combine terms and add bias
        scores = bilinear_term + linear_dep_term.unsqueeze(3) + linear_head_term.unsqueeze(2) + self.b.view(1, -1, 1, 1)
        
        # Permute to get shape [batch, seq_len, seq_len, out_features] and remove last dim if it's 1
        return scores.permute(0, 2, 3, 1).squeeze(-1)

class BiaffineParser(nn.Module):
    def __init__(self, encoder, hidden_size=768, arc_mlp_size=500, label_mlp_size=100, n_labels=50):
        super().__init__()
        self.encoder = encoder
        
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

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(outputs.last_hidden_state)

        # Apply MLPs to get representations for head and dependent
        arc_h = self.relu(self.arc_mlp_head(h))
        arc_d = self.relu(self.arc_mlp_dep(h))
        label_h = self.relu(self.label_mlp_head(h))
        label_d = self.relu(self.label_mlp_dep(h))
        
        # Dropout on MLP outputs
        arc_h = self.dropout(arc_h)
        arc_d = self.dropout(arc_d)
        label_h = self.dropout(label_h)
        label_d = self.dropout(label_d)

        # Get scores from biaffine classifiers
        # arc_scores: [batch, seq_len, seq_len]
        arc_scores = self.arc_biaffine(arc_d, arc_h)
        
        # label_scores: [batch, seq_len, seq_len, n_labels]
        label_scores = self.label_biaffine(label_d, label_h)
        
        return arc_scores, label_scores