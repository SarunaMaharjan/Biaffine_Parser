# src/models_rs.py
import torch
from torch import nn
from transformers import RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaOutput, RobertaIntermediate, RobertaSelfOutput

# ... (StructInformedAttention, StructInformedRobertaLayer, CNNParser, StructRoberta, Biaffine classes remain the same) ...
# ... (Paste the unchanged classes here) ...


class StructInformedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.struct_gate = nn.Parameter(torch.zeros(self.num_attention_heads) + 0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, struct_attn_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if struct_attn_mask is not None:
            gate = self.struct_gate.view(1, -1, 1, 1)
            attention_scores = attention_scores + (struct_attn_mask.unsqueeze(1) * gate)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return (context_layer,)

class StructInformedRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = StructInformedAttention(config)
        self.attention_output = RobertaSelfOutput(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None, struct_attn_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, struct_attn_mask=struct_attn_mask)
        attention_output = self.attention_output(self_attention_outputs[0], hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,)

class CNNParser(nn.Module):
    def __init__(self, hidden_size, num_parser_layers=3, kernel_size=3):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(num_parser_layers)
        ])
        self.head_predictor = nn.Linear(hidden_size, hidden_size)
        self.child_predictor = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        x = hidden_states.transpose(1, 2)
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        x = x.transpose(1, 2)
        head_rep = self.head_predictor(x)
        child_rep = self.child_predictor(x)
        dependency_scores = torch.einsum('bic,bjc->bij', child_rep, head_rep)
        return dependency_scores

class StructRoberta(nn.Module):
    def __init__(self, model_name="roberta-base", num_front_layers=6):
        super().__init__()
        # Load config, not the full model, to build the architecture
        base_model = RobertaForMaskedLM.from_pretrained(model_name)
        self.config = base_model.roberta.config
        self.embeddings = base_model.roberta.embeddings
        all_layers = base_model.roberta.encoder.layer
        
        self.front_layers = nn.ModuleList(all_layers[:num_front_layers])
        self.rear_layers = nn.ModuleList()
        if len(all_layers) > num_front_layers:
            for i in range(num_front_layers, len(all_layers)):
                self.rear_layers.append(StructInformedRobertaLayer(self.config))

        self.parser = CNNParser(self.config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
             attention_mask = (input_ids != self.config.pad_token_id).float()
        
        # This function is part of the original RobertaModel, we need to replicate it
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        hidden_states = self.embeddings(input_ids=input_ids)
        for layer in self.front_layers:
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
        
        struct_scores = self.parser(hidden_states)
        
        rear_hidden_states = hidden_states
        if self.rear_layers:
            for layer in self.rear_layers:
                layer_outputs = layer(
                    rear_hidden_states,
                    attention_mask=extended_attention_mask,
                    struct_attn_mask=struct_scores,
                )
                rear_hidden_states = layer_outputs[0]
        
        # Return the final hidden states for the parser head
        return rear_hidden_states

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

### --- THE FIX IS IN THIS CLASS ---
class StructBiaffineParser(nn.Module):
    # MODIFIED: Removed hardcoded hidden_size from the arguments
    def __init__(self, model_name="roberta-base", num_front_layers=6,
                 arc_mlp_size=500, label_mlp_size=100, n_labels=50):
        super().__init__()
        self.encoder = StructRoberta(model_name=model_name, num_front_layers=num_front_layers)
        
        # --- START OF FIX ---
        # Dynamically get the hidden size from the loaded encoder's configuration
        hidden_size = self.encoder.config.hidden_size
        print(f"Initializing parser head with hidden size: {hidden_size}")
        # --- END OF FIX ---

        # MLPs for arc prediction (now uses the correct hidden_size)
        self.arc_mlp_head = nn.Linear(hidden_size, arc_mlp_size)
        self.arc_mlp_dep = nn.Linear(hidden_size, arc_mlp_size)
        
        # MLPs for label prediction (now uses the correct hidden_size)
        self.label_mlp_head = nn.Linear(hidden_size, label_mlp_size)
        self.label_mlp_dep = nn.Linear(hidden_size, label_mlp_size)
        
        # Biaffine classifiers
        self.arc_biaffine = Biaffine(arc_mlp_size, 1)
        self.label_biaffine = Biaffine(label_mlp_size, n_labels)
        
        self.dropout = nn.Dropout(0.33)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from the custom StructRoberta encoder
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(h)

        # Apply MLPs for biaffine parser head
        arc_h = self.relu(self.arc_mlp_head(h))
        arc_d = self.relu(self.arc_mlp_dep(h))
        label_h = self.relu(self.label_mlp_head(h))
        label_d = self.relu(self.label_mlp_dep(h))
        
        arc_h, arc_d = self.dropout(arc_h), self.dropout(arc_d)
        label_h, label_d = self.dropout(label_h), self.dropout(label_d)

        arc_scores = self.arc_biaffine(arc_d, arc_h)
        label_scores = self.label_biaffine(label_d, label_h)
        
        return arc_scores, label_scores