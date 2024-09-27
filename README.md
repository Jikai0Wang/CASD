# TRAG: Token-level Retrieval-augmented Generation Through Speculative Decoding

## Contents
- [Installation](#installation)
- [Demo](#Demo)
  - [With independent draft models](#With-independent-draft-models)
  - [With EAGLE draft models](#With-EAGLE-draft-models)
- [Evaluation on datasets](#Evaluation-on-datasets)
  - [With independent draft models](#With-independent-draft-models)
  - [With EAGLE draft models](#With-EAGLE-draft-models)
- [Citation](#citation)


## Installation
```bash
#Recommended Python version >= 3.9
pip install -r requirements.txt
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cd DraftRetriever
maturin build --release --strip -i python3.9 # will produce a .whl file in target/
pip install [.whl]
```

## Evaluation
### Baseline
```bash
export CUDA_VISIBLE_DEVICES=0 #Also support for multiple GPUs
python pred.py
```
### TRAG
```bash
export CUDA_VISIBLE_DEVICES=0 #Also support for multiple GPUs
python pred_trag.py --threshold 0.1 --k 0
```

## Evaluation with LLM-Lingua2
### Baseline
```bash
export CUDA_VISIBLE_DEVICES=0 #Also support for multiple GPUs
python pred_lingua.py --compress_rate 0.33
```
### TRAG
```bash
export CUDA_VISIBLE_DEVICES=0
python pred_trag_lingua.py --compress_rate 0.33 --threshold 0.1 --k 0
```

## Calculate Scores
```bash
python cal_score.py --path [output path]
```

Some of the code in this project refers to [Longbench](https://github.com/THUDM/LongBench) and [REST](https://github.com/FasterDecoding/REST).
