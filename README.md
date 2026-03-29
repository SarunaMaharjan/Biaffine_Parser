# Biaffine Parser for Hindi Dependency Parsing


## Overview

This repository implements a graph-based biaffine dependency parser 
for Hindi, fine-tuned on top of four transformer language models:

- **XLMR** — XLM-RoBERTa baseline
- **Struct_XLMR** — Structure-aware XLM-RoBERTa with CNN parser
- **Roberta_hi** — Hindi-pretrained RoBERTa baseline
- **Struct_Roberta_hi** — Structure-aware Hindi RoBERTa with CNN parser

The biaffine parser implementation is adapted from 
[chantera/biaffineparser](https://github.com/chantera/biaffineparser).  
The StructFormer architectural modifications are adapted from 
[yikangshen/UDGN](https://github.com/yikangshen/UDGN).

---

## Repository Structure
```
Biaffine_Parser/
├── Roberta_scripts/         # Fine-tuning scripts for Roberta_hi
├── Struct_Roberta_scripts/  # Fine-tuning scripts for Struct_Roberta_hi
├── Struct_XLMR_folder/      # Fine-tuning scripts for Struct_XLMR
├── XLMR_folder/             # Fine-tuning scripts for XLMR
```

---

## Requirements
Python 3.8+
```bash
pip install transformers torch datasets accelerate
```

---

## Data

Fine-tuning uses the **Hindi Universal Dependencies Treebank (HDTB)**  
available at: https://github.com/UniversalDependencies/UD_Hindi-HDTB

The treebank is in CoNLL-U format and is split into:
- ~13k sentences for training
- ~1.6k sentences for development
- ~1.6k sentences for testing

---

## Usage

### Fine-tuning script example
```bash
cd specific_folder
bash run_training.sh
bash run_evaluation.sh
```


---

## Results

| Model | UAS (%) | LAS (%) |
|---|---|---|
| XLMR | 71.62 | 69.96 |
| Struct_XLMR | **91.70** | **84.13** |
| Roberta_hi | 68.60 | 62.48 |
| Struct_Roberta_hi | 75.64 | 69.07 |

Evaluated on the HDTB test set using standard UAS and LAS metrics.

---



## Acknowledgements

- Biaffine parser adapted from [chantera/biaffineparser](https://github.com/chantera/biaffineparser)
- StructFormer architecture adapted from [yikangshen/UDGN](https://github.com/yikangshen/UDGN)
- Hindi UD Treebank: [UD_Hindi-HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB)
