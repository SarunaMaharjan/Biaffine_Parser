# src/eval.py
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, AutoTokenizer
from models_std import BiaffineParser # Assuming you've renamed the files as suggested
from data_std import DependencyDataset, get_collate_fn
from main_std import evaluate
import json # ADDED: For loading the label map
import os   # ADDED: For handling file paths

def main():
    parser = argparse.ArgumentParser()
    # MODIFIED: Changed to model_dir to load both the model and the label map
    parser.add_argument('--model_dir', required=True, default="src/XLMR_parser", help="Directory containing the saved model and label2id.json")
    parser.add_argument('--data', required=True, help="Path to the test .conllu file")
    parser.add_argument('--pretrained_model', default='xlm-roberta-base-local', help="Base model identifier for architecture")
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_prefix_space=True)

    # --- START OF FIX ---
    # 1. Load the label map created during training
    label_map_path = os.path.join(args.model_dir, 'label2id.json')
    with open(label_map_path, 'r') as f:
        label2id = json.load(f)
    n_labels = len(label2id)
    print(f"Loaded label map with {n_labels} labels from training.")

    # 2. Pass the loaded map to the dataset to ensure consistency
    test_data = DependencyDataset(args.data, tokenizer, label2id=label2id)
    # --- END OF FIX ---
    
    collate_fn_with_tokenizer = get_collate_fn(tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn_with_tokenizer)
    
    # 3. Initialize model architecture with the correct number of labels from the loaded map
    encoder = XLMRobertaModel.from_pretrained(args.pretrained_model)
    model = BiaffineParser(encoder, n_labels=n_labels).to(device)

    # Load the fine-tuned weights
    model_path = os.path.join(args.model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # Evaluate on the test set
    uas, las = evaluate(model, test_loader, device)
    print("\n--- Final Test Results ---")
    print(f"UAS: {uas:.2f}%")
    print(f"LAS: {las:.2f}%")
    print("--------------------------")

if __name__ == "__main__":
    main()
