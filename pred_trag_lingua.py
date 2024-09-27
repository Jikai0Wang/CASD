import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np
import random
import argparse
from utils.test import rest_forward
from rest.model.rest_model import RestModel
import draftretriever


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct-evals")
    parser.add_argument('--compress_rate', type=float, default=0.33)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=0)
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt):
    messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": prompt},]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return prompt

def post_process(response):
    response = response.replace("assistant\n\n", "")
    return response


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset,
             out_path, num_draft=64, max_token_span=16, threshold=0.1, k=10):
    id=0
    for json_obj in tqdm(data):
        id+=1
        prompt=json_obj["lingua_prompt"]
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(torch.device('cuda:0'))
        context_length = input.input_ids.shape[-1]

        token_spans = list(range(2, max_token_span + 1))[::-1]

        writer = draftretriever.Writer(
            index_file_path=f'tmp/lingua0.33_{k}_{threshold}.idx',
            vocab_size=tokenizer.vocab_size
        )
        if "ori_context" in json_obj:
            document = tokenizer(json_obj["ori_context"], truncation=False).input_ids
        else:
            document = tokenizer(json_obj["context"], truncation=False).input_ids
        writer.add_entry(document[1:])

        writer.finalize()
        datastore = draftretriever.Reader(
            index_file_path=f'tmp/lingua0.33_{k}_{threshold}.idx',
        )
        try:
            with torch.no_grad():
                output_ids, new_token, idx, accept_length_tree, start_time = rest_forward(
                    input.input_ids,
                    model,
                    tokenizer,
                    max_gen,
                    0,
                    0,
                    datastore,
                    num_draft,
                    token_spans,
                    threshold,
                    k,
                )
        except:
            continue

        pred = tokenizer.decode(output_ids[0, input.input_ids.size(1):], skip_special_tokens=True)
        pred = post_process(pred)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"id":id,"pred": pred, "answers": json_obj["answers"],
                        "context_length": context_length, "new_token": new_token.tolist(), "step": idx + 1}, f,
                      ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    model = RestModel.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    threshold = args.threshold
    k=args.k

    max_length = 8192
    datasets = ["nq","triviaqa", "2wikimqa","hotpotqa", "multi_news", "gov_report", "lcc", "repobench-p"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    output_path=f"pred_lingua_{args.compress_rate}_trag_"+str(threshold)
    if k>0:
        output_path=output_path+f"_top{k}"

    model, tokenizer = load_model_and_tokenizer(args.model)
    for dataset in datasets:
        print(f"Running {dataset} ...")
        data=[]
        data_path=f"data/lingua_{args.compress_rate}/{dataset}.json"
        with open(data_path, "r") as f:
            for i in f.readlines():
                data.append(json.loads(i.strip()))

        if not os.path.exists(f"{output_path}"):
            os.makedirs(f"{output_path}")
        out_path = f"{output_path}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        get_pred(model, tokenizer, data, max_length, \
                                                  max_gen, prompt_format, dataset,
                                                  out_path, 64,
                                                  16, threshold,k)

