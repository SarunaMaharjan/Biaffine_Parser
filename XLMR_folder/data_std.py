# src/data.py
import torch
from torch.utils.data import Dataset
from functools import partial

class DependencyDataset(Dataset):
    # MODIFIED: Added 'label2id=None' to accept the label map
    def __init__(self, file_path, tokenizer, label2id=None):
        self.tokenizer = tokenizer
        self.sentences = []
        
        # If a label2id map is not provided, we need to build one.
        if label2id is None:
            build_labels = True
            self.labels = set(['<pad>']) 
        else:
            build_labels = False

        with open(file_path, 'r', encoding='utf-8') as f:
            words, heads, labels = ['[ROOT]'], [0], ['_']
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    if len(words) > 1:
                        self.sentences.append({'words': words, 'heads': heads, 'labels': labels})
                    words, heads, labels = ['[ROOT]'], [0], ['_']
                else:
                    parts = line.strip().split('\t')
                    if '-' in parts[0] or '.' in parts[0]: continue
                    words.append(parts[1])
                    heads.append(int(parts[6]))
                    current_label = parts[7]
                    labels.append(current_label)
                    if build_labels:
                        self.labels.add(current_label)
        
        if len(words) > 1:
            self.sentences.append({'words': words, 'heads': heads, 'labels': labels})

        # If a label map was passed in, use it. Otherwise, create the one we built.
        if label2id is not None:
            self.label2id = label2id
        else:
            self.label2id = {label: i for i, label in enumerate(sorted(list(self.labels)))}
        
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(self.label2id)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = sentence['words']
        heads = sentence['heads']
        # Use .get() to handle labels that might be in dev/test but not train
        labels = [self.label2id.get(l, 0) for l in sentence['labels']] 

        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=True, padding=False)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        word_ids = encoding.word_ids()
        aligned_heads = [-100] * len(word_ids)
        aligned_labels = [-100] * len(word_ids)
        
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                aligned_heads[i] = heads[word_idx]
                aligned_labels[i] = labels[word_idx]
            previous_word_idx = word_idx
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'heads': torch.tensor(aligned_heads, dtype=torch.long),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

# This part can remain the same
def collate_fn(batch, pad_token_id):
    max_len = max(len(item['input_ids']) for item in batch)

    padded_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_heads = torch.full((len(batch), max_len), -100, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        padded_input_ids[i, :seq_len] = item['input_ids']
        padded_attention_mask[i, :seq_len] = item['attention_mask']
        padded_heads[i, :seq_len] = item['heads']
        padded_labels[i, :seq_len] = item['labels']

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'heads': padded_heads,
        'labels': padded_labels
    }

def get_collate_fn(tokenizer):
    return partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
