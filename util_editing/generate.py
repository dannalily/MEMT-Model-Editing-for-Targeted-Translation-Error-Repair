from typing import List
import unicodedata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def input_format(
        tok: AutoTokenizer,
        prompts: List[str],
        system_prompt: str,
        chat_temp: bool = False):
    
    if chat_temp:
        chat_texts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            chat_texts.append(
                tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return chat_texts
    else:
        return prompts

    

def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    max_out_len: int = 200,
    temperature: float = 1.1,
    top_p: float = 0.9,
    top_k: int = 5,
    repetition_penalty: float = 1.08,
    no_repeat_ngram_size: int = 3,
    system_prompt: str = "You are a helpful assistant. Just give me the answer without anything else.",
    vanilla_generation: bool = False,
    chat_temp: bool = False,
) -> List[str]:

    device = next(model.parameters()).device
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    expanded_prompts = [p for p in prompts for _ in range(n_gen_per_prompt)]

    if chat_temp:
        chat_texts = []
        for p in expanded_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            chat_texts.append(
                tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
    else:
        chat_texts = expanded_prompts

    model_inputs = tok(chat_texts, return_tensors="pt", padding=True).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_out_len,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    if vanilla_generation:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ))

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **gen_kwargs)

    input_ids = model_inputs["input_ids"]
    generations = []
    for in_ids, out_ids in zip(input_ids, outputs):
        new_tokens = out_ids[len(in_ids):]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        text = unicodedata.normalize("NFKD", text).replace("\n\n", " ").strip()
        generations.append(text)

    return generations
