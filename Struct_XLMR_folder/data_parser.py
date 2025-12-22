# data_parser.py (FINAL REPAIRED VERSION)
import torch
from torch.utils.data import Dataset
from functools import partial

class DependencyDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id=None):
        self.tokenizer = tokenizer
        self.sentences = []
        
        if label2id is None:
            build_labels = True
            self.labels = set(['<pad>', 'ROOT', '_']) # <pad> will be removed
        else:
            build_labels = False
            self.labels = set() # Not used if label2id is provided

        # CoNLL-U format indices
        ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

        with open(file_path, 'r', encoding='utf-8') as f:
            words, heads, labels = ['[ROOT]'], [0], ['ROOT']
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    if len(words) > 1:
                        self.sentences.append({'words': words, 'heads': heads, 'labels': labels})
                    words, heads, labels = ['[ROOT]'], [0], ['ROOT']
                else:
                    parts = line.split('\t')
                    if '-' in parts[ID] or '.' in parts[ID]: continue
                    
                    words.append(parts[FORM])
                    try:
                        heads.append(int(parts[HEAD]))
                        current_label = parts[DEPREL]
                    except ValueError:
                        continue

                    labels.append(current_label)
                    if build_labels:
                        self.labels.add(current_label)
        
        if len(words) > 1:
            self.sentences.append({'words': words, 'heads': heads, 'labels': labels})
            
        if label2id is not None:
            self.label2id = label2id
        else:
            # --- START: FINAL BUG FIX ---
            # 1. Remove '<pad>' from the set *before* sorting and indexing
            self.labels.discard('<pad>') 
            
            # 2. Now sort the clean list
            sorted_labels = sorted(list(self.labels))
            if 'ROOT' in sorted_labels:
                sorted_labels.remove('ROOT')
                sorted_labels.insert(0, 'ROOT')
            
            # 3. Create the map. Indices will now be 0 to (num_labels - 1)
            self.label2id = {label: i for i, label in enumerate(sorted_labels)}
            # --- END: FINAL BUG FIX ---
            
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(self.label2id) # This will be 27
        # The max index in label2id will now be 26. The bug is fixed.

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = sentence['words']
        heads = sentence['heads']
        
        # Use .get() for safety, defaulting unknown labels to 'ROOT' (index 0)
        labels = [self.label2id.get(l, self.label2id.get('ROOT', 0)) for l in sentence['labels']] 

        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=True, padding=False)
        input_ids, attention_mask = encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
        
        word_ids = encoding.word_ids()
        aligned_heads, aligned_labels = [-100] * len(word_ids), [-100] * len(word_ids)
        
        word_idx_to_first_token_idx = {}
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_idx_to_first_token_idx:
                word_idx_to_first_token_idx[word_idx] = i
        
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None: continue
            
            if word_idx != previous_word_idx and word_idx > 0:
                gold_head_word_idx = heads[word_idx]
                aligned_heads[i] = word_idx_to_first_token_idx.get(gold_head_word_idx, -100)
                aligned_labels[i] = labels[word_idx]
            
            if word_idx == 0:
                aligned_heads[i], aligned_labels[i] = -100, -100 
            
            previous_word_idx = word_idx
        
        return { 'input_ids': input_ids, 'attention_mask': attention_mask,
                 'heads': torch.tensor(aligned_heads, dtype=torch.long),
                 'labels': torch.tensor(aligned_labels, dtype=torch.long) }

def collate_fn(batch, pad_token_id):
    max_len = max(len(item['input_ids']) for item in batch)
    padded_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_heads = torch.full((len(batch), max_len), -100, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        padded_input_ids[i, :seq_len], padded_attention_mask[i, :seq_len] = item['input_ids'], item['attention_mask']
        padded_heads[i, :seq_len], padded_labels[i, :seq_len] = item['heads'], item['labels']

    # This clamp is still a good safety net, so we keep it.
    seq_lens = padded_attention_mask.sum(dim=1)
    for i in range(len(batch)):
        padded_heads[i, padded_heads[i] >= seq_lens[i]] = -100
        padded_heads[i, padded_heads[i] < 0] = -100

    return { 'input_ids': padded_input_ids, 'attention_mask': padded_attention_mask,
             'heads': padded_heads, 'labels': padded_labels }

def get_collate_fn(tokenizer):
    return partial(collate_fn, pad_token_id=tokenizer.pad_token_id)