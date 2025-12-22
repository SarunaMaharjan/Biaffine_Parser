import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class StructXLMRParser(nn.Module):
    def __init__(self, backbone, hidden_size, num_labels_arc, num_labels_rel, dropout=0.33):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.arc_head = nn.Linear(hidden_size, num_labels_arc)
        self.arc_dep = nn.Linear(hidden_size, num_labels_arc)
        self.label_head = nn.Linear(hidden_size, num_labels_rel)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_pretrained(cls, checkpoint_path, num_labels_arc, num_labels_rel):
        backbone = XLMRobertaModel.from_pretrained(checkpoint_path)
        hidden_size = backbone.config.hidden_size
        model = cls(backbone, hidden_size, num_labels_arc, num_labels_rel)
        return model

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = self.dropout(outputs.last_hidden_state)
        arc_head_scores = self.arc_head(seq_output)
        arc_dep_scores = self.arc_dep(seq_output)
        label_scores = self.label_head(seq_output)
        return arc_head_scores, arc_dep_scores, label_scores
