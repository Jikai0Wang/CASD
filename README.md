# CASD: Enhancing Generation Accuracy via Context-Aware Speculative Decoding.

## Contents
- [Installation](#installation)
- [Evaluation](#evaluation)
  - [Baseline](#baseline)
  - [CASD](#casd)
- [Evaluation with LLM-Lingua2](#evaluation-with-llm-lingua2)
  - [Baseline](#baseline)
  - [CASD](#casd)
- [Calculate Scores](#calculate-scores)


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
### CASD
```bash
export CUDA_VISIBLE_DEVICES=0 #Also support for multiple GPUs
python pred_casd.py --threshold 0.1 --k 0
```

## Evaluation with LLM-Lingua2
### Baseline
```bash
export CUDA_VISIBLE_DEVICES=0 #Also support for multiple GPUs
python pred_lingua.py --compress_rate 0.33
```
### CASD
```bash
export CUDA_VISIBLE_DEVICES=0
python pred_casd_lingua.py --compress_rate 0.33 --threshold 0.1 --k 0
```

## Calculate Scores
```bash
python cal_score.py --path [output path]
```

Some of the code in this project refers to [Longbench](https://github.com/THUDM/LongBench) and [REST](https://github.com/FasterDecoding/REST).
