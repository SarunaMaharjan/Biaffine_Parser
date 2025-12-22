# eval_simple.py
import argparse, os, torch, json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Import the correct model and data classes
from struct_xlmr import StructBiaffineParser
from data_parser import DependencyDataset, get_collate_fn

# --- Import the 'evaluate' function from main_parser.py ---
# This is the simple argmax evaluation, NOT the MST one.
from main_parser import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help="Directory containing the saved model and label2id.json")
    parser.add_argument('--data', required=True, help="Path to the test .conllu file")
    parser.add_argument('--base_model_name', default='src/models/xlm-roberta-base-local', help="Base model name for architecture config")
    parser.add_argument('--tokenizer_path', required=True, help="Path to the tokenizer directory")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_prefix_space=True)

    label_map_path = os.path.join(args.model_dir, 'label2id.json')
    with open(label_map_path, 'r') as f: label2id = json.load(f)
    n_labels = len(label2id)
    print(f"Loaded label map with {n_labels} labels.")

    test_data = DependencyDataset(args.data, tokenizer, label2id=label2id)
    collate_fn_with_tokenizer = get_collate_fn(tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn_with_tokenizer)
    
    # Initialize the custom model architecture
    print(f"Loading model architecture from {args.base_model_name}...")
    model = StructBiaffineParser(model_name=args.base_model_name, n_labels=n_labels).to(device)

    # Load the fine-tuned weights
    model_path = os.path.join(args.model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # Call the imported 'evaluate' function
    uas, las = evaluate(model, test_loader, device)
    
    print("\n--- Final Test Results (Argmax Decoding) ---")
    print(f"UAS: {uas:.2f}%")
    print(f"LAS: {las:.2f}%")
    print("--------------------------------------------")

if __name__ == "__main__":
    main()
