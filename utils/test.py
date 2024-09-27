import argparse
import json
import os
import random
import time
import shortuuid
import torch
import numpy as np
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions
from fastchat.model import load_model, get_conversation_template

# Rest imports
import transformers

import sys
sys.path.append("../")

from rest.model.utils import *
from rest.model.rest_model import RestModel
from rest.model.kv_cache import initialize_past_key_values
import draftretriever

@torch.inference_mode()
def rest_forward(input_ids, model, tokenizer, max_new_token, temperature, top_p, datastore, num_draft, token_spans, threshold,k,
                 max_steps=1024):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len + 1
    model.base_model.model.draft_mask = None
    logits = initialize_logits(
        input_ids, model, past_key_values
    )
    new_token = 0

    torch.cuda.synchronize()
    start_time = time.time()
    for idx in range(max_steps):
        candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
            logits,
            input_ids,
            datastore,
            token_spans,
            top_p,
            temperature,
            max_num_draft=num_draft,
            device=model.base_model.device
        )
        model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            draft_buffers["draft_position_ids"],
            input_ids,
            draft_buffers["retrieve_indices"],
        )
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, tokenizer.eos_token_id, threshold, k,
        )
        input_ids, logits, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            draft_buffers["retrieve_indices"],
            outputs,
            logits,
            new_token,
            past_key_values_data,
            current_length_data,
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_token:
            break
    return input_ids, new_token, idx, accept_length_list, start_time
