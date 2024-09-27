import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np
import random
import argparse
import draftretriever


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct-evals")
    parser.add_argument('--compress_rate', type=float, default=0.33)
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
             out_path):
    id=0
    for json_obj in tqdm(data):
        id+=1
        prompt=json_obj["lingua_prompt"]

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(torch.device('cuda:0'))
        context_length = input.input_ids.shape[-1]

        output = model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            pad_token_id=128004,
            eos_token_id=tokenizer.eos_token_id
        )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"id":id,"pred": pred, "answers": json_obj["answers"],
                        "context_length": context_length}, f,
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
    model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16,device_map="auto")
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    max_length = 8192
    datasets = ["nq","triviaqa", "2wikimqa","hotpotqa", "multi_news", "gov_report", "lcc", "repobench-p"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    output_path=f"pred_lingua_{args.compress_rate}"

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
                                                  out_path)

