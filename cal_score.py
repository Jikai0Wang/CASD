import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_score,
    code_sim_score,
)

dataset2metric = {
    "nq": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "triviaqa": qa_f1_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="pred")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset == "triviaqa":
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        scores.append(score)

    scores = round(100 * np.mean(scores), 2)
    return scores


if __name__ == '__main__':
    args = parse_args()
    scores = dict()

    path = f"{args.path}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    sum=0
    c=0
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = None
                if "length" in data:
                    lengths.append(data["length"])
        score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        scores[dataset] = score
        sum+=score
        c+=1
    scores["avg"]=round(sum/c,2)

    out_path = f"{path}/score.json"

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
