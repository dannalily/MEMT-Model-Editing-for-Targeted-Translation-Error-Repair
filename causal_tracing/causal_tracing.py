import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re
import argparse
from collections import defaultdict

import numpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util_editing import nethook
from tqdm import tqdm
from utils import *        # expects LANGS, read_json_file, etc.
from prompt import PROMPT_DICT


# ---------------------------
# Helpers
# ---------------------------

def layername(model, num, kind=None):
    if kind == "embed":
        return "model.embed_tokens"
    return f'model.layers.{num}{"" if kind is None else "." + kind}'


def concat_dicts(d1, d2):
    return {
        "input_ids": torch.cat([d1["input_ids"], d2["input_ids"]], dim=1),
        "attention_mask": torch.cat([d1["attention_mask"], d2["attention_mask"]], dim=1),
    }


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def make_inputs(tokenizer, prompts, device="cuda"):
    # Ensure we have a pad token for causal models
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in enc.items()}


def compute_base_score(model, tokenizer, prompt, correct_answer):
    with torch.no_grad():
        enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        enc_answer = tokenizer(correct_answer, return_tensors="pt", add_special_tokens=False).to(model.device)

        concat_ids = torch.cat([enc_prompt.input_ids, enc_answer.input_ids], dim=1)
        concat_mask = torch.cat([enc_prompt.attention_mask, enc_answer.attention_mask], dim=1)

        logits = model(input_ids=concat_ids, attention_mask=concat_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        ans_ids = enc_answer.input_ids[0]
        prompt_len = enc_prompt.input_ids.size(1)

        token_logps = log_probs[0, prompt_len-1:-1].gather(1, ans_ids.unsqueeze(1)).squeeze(1)
        avg_log_prob = token_logps.mean()

    return avg_log_prob


def find_token_range(tokenizer, fullstring: str, substring: str):
    """Robustly align substring to token indices using offset mapping."""
    enc = tokenizer(fullstring, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    start_char = fullstring.find(substring)
    if start_char == -1:
        raise ValueError("substring not found in fullstring")
    end_char = start_char + len(substring)

    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if tok_start is None and s <= start_char < e:
            tok_start = i
        if tok_start is not None and e >= end_char:
            tok_end = i + 1
            break
    if tok_start is None or tok_end is None:
        raise ValueError("could not align substring to token boundaries")
    return tok_start, tok_end


def trace_with_patch(
    model,                 # The model
    inp,                   # A set of inputs
    states_to_patch,       # A list of (token index, layername) pairs to restore
    answers_t,             # Answer tokenized tensors
    tokens_to_mix,         # Range of tokens to corrupt (begin, end)
    noise=0.1,             # Level of noise to add
    trace_layers=None,     # List of traced outputs to return
    seed=1,                # Reproducibility
):
    torch.manual_seed(seed)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        # x is the hidden state at this layer
        if layer == embed_layername:
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                h = x
                if h.shape[0] > 1 and e > b:
                    noise_tensor = torch.randn(
                        (h.shape[0] - 1, e - b, h.shape[2]),
                        device=h.device,
                        dtype=h.dtype,
                    ) * noise
                    h[1:, b:e] = h[1:, b:e] + noise_tensor
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    additional_layers = [] if trace_layers is None else trace_layers
    merged = concat_dicts(inp, answers_t)
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**merged)  # [batch, seq(prompt+answer), Vocab]

    # Collect softmax probabilities for the answers_t token predictions of interest.
    P = inp["input_ids"].shape[1]
    A = answers_t["input_ids"].shape[1]
    logits = outputs_exp.logits
    logp = torch.log_softmax(logits, dim=-1)

    window = logp[:, P-1:P+A-1, :]          # [B, A, V]
    y = answers_t["input_ids"]              # [B, A]
    tok_logp = window.gather(2, y.unsqueeze(-1)).squeeze(-1)  # [B, A]
    seq_logp = tok_logp.mean(dim=1)         # [B]
    probs = seq_logp.mean()                 # scalar

    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_important_states(model, num_layers, inp, e_range, answer_t, low, noise=0.1, start_index=0):
    ntoks = inp["input_ids"].shape[1]
    search_start, _ = e_range
    table = []
    # Pre-fill before search_start so tensor shapes align in visualization
    for _tnum in range(max(0, start_index), search_start):
        row = []
        for _layer in range(0, num_layers):
            row.append(torch.tensor(low, device=inp["input_ids"].device))
        table.append(torch.stack(row))

    for tnum in tqdm(range(search_start, ntoks)):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, low, window=10, noise=0.1, start_index=0
):
    ntoks = inp["input_ids"].shape[1]
    search_start, _ = e_range
    table = []
    # Pre-fill area before search window
    for _tnum in range(max(0, start_index), search_start):
        row = []
        for _layer in range(0, num_layers):
            row.append(torch.tensor(low, device=inp["input_ids"].device))
        table.append(torch.stack(row))

    for tnum in tqdm(range(search_start, ntoks)):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(max(0, layer - window // 2), min(num_layers, layer - (-window // 2)))
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def collect_embedding_std(mt, subjects, batch_size=16):
    tokenizer = mt["tokenizer"]
    model = mt["model"]

    outs = []
    for i in range(0, len(subjects), batch_size):
        batch = subjects[i:i + batch_size]
        inp = make_inputs(tokenizer, batch, device=model.device)
        with nethook.Trace(model, layername(model, 0, "embed")) as t:
            with torch.no_grad():
                model(**inp)
            outs.append(t.output[0])  # [B, T, H]
    alldata = torch.cat(outs, dim=0)
    noise_level = alldata.std().item()
    return noise_level


def ensure_dir_for_file(filename: str):
    """Ensure directory for filename exists."""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ---------------------------
# Main causal tracing routine
# ---------------------------

def calculate_hidden_flow(
    mt, prompt, subject, correct_answer, samples=10, noise=0.1, window=10, kind=None,user_start_index=0
):
    """
    Runs causal tracing over every token/layer combination and returns a dictionary of results.
    """
    tokenizer = mt["tokenizer"]
    model = mt["model"]

    base_score = compute_base_score(model, tokenizer, prompt, correct_answer)

    # Token span of the subject in the full prompt (aligned with add_special_tokens=False)
    e_range = find_token_range(tokenizer, prompt, subject)

    inp = make_inputs(tokenizer, [prompt] * (samples + 1), device=model.device)
    answer_t = make_inputs(tokenizer, [correct_answer] * (samples + 1), device=model.device)

    low_score = trace_with_patch(
        model, inp, [], answer_t, e_range, noise=noise
    ).item()

    if not kind:
        differences = trace_important_states(
            model, mt["num_layers"], inp, e_range, answer_t, noise=noise,  low=low_score
        )
    else:
        differences = trace_important_window(
            model,
            mt["num_layers"],
            inp,
            e_range,
            answer_t,
            kind=kind,
            low=low_score,
            window=window,
            noise=noise,
        )
    differences = differences.detach().cpu()

    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0].cpu(),
        input_tokens=decode_tokens(tokenizer, inp["input_ids"][0].cpu()),
        subject_range=e_range,
        answer=correct_answer,
        window=window,
        kind=kind or "",
        user_start_index=user_start_index,
    )


# ---------------------------
# Entry point
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translator", type=str, choices=['Qwen2-5-3B-Instruct', 'Qwen2-5-7B-Instruct'])
    parser.add_argument("--slang", type=str, required=True, help="source lang, en, zh")
    parser.add_argument("--tlang", type=str, required=True, help="target lang, en, zh, ja, fr, de, ar")
    args = parser.parse_args()

    correct_data_file = f"MEMT/Keep_Samples/{args.translator}/{args.slang}_{args.tlang}.json"
    idiom_span_file = f"MEMT/idiom_span_{args.slang}2x.json"

    correct_data = read_json_file(correct_data_file)
    idiom_span_dict = read_json_file(idiom_span_file)

    if args.translator == 'Qwen2-5-3B-Instruct':
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    elif args.translator == 'Qwen2-5-7B-Instruct':
        model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model (single GPU: remove device_map and do .to("cuda"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    ).to("cuda")

    nethook.set_requires_grad(False, model)
    model.eval()

    # Discover layer count
    layer_names = [
        n
        for n, _ in model.named_modules()
        if re.match(r"^(?:model\.)?layers\.\d+$", n)
    ]
    num_layers = len(layer_names)

    samples = 10
    window = 10
    kinds = [None, "self_attn", "mlp"]

    mt = {
        "model": model,
        "tokenizer": tokenizer,
        "num_layers": num_layers
    }

    # Subjects for noise calibration
    subjects = [idiom_span_dict[str(item["id"])] for item in correct_data]
    noise_level = 3 * collect_embedding_std(mt, subjects)
    print(f"Using noise level {noise_level}")

    for kind in kinds:
        kind_suffix = f"_{kind}" if kind else ""
        for i, item in enumerate(correct_data[:100]):
            print(f"{i}/{len(correct_data)}")

            system_prompt = PROMPT_DICT['GENERAL_TRANS']['sys'].format(SLang=LANGS[args.slang], TLang=LANGS[args.tlang])
            user_prompt = PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang=LANGS[args.slang], Source=item["src"], TLang=LANGS[args.tlang])

            correct_answer = item['correct_trans'][LANGS[args.tlang]]["trans"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            # Chat template -> plain text prompt (no extra specials at tokenization time)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            subject = idiom_span_dict[str(item["id"])]
            if subject not in prompt:
                raise ValueError(f"Subject span not present in prompt for item id {item['id']}.")

            result = calculate_hidden_flow(
                mt, prompt, subject, correct_answer, samples=samples, noise=noise_level, window=window, kind=kind
            )

            # Convert tensors to numpy for saving; keep non-tensors as-is
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }

            file_name = f"Causal_Tracing_Results/{args.translator}/{args.slang}_{args.tlang}/index_{i}{kind_suffix}.npz"
            ensure_dir_for_file(file_name)
            numpy.savez(file_name, **numpy_result)


if __name__ == "__main__":
    main()
