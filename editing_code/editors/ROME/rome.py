from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List, Tuple
from copy import deepcopy
from rome_hparams import ROMEHyperParams
import torch
from util_editing import nethook
from util_editing.generate import generate_fast
from editors.ROME.compute_u import compute_u
from editors.ROME.compute_v import compute_v
import gc
CONTEXT_TEMPLATES_CACHE = None

def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    system_prompt="",
    random_context_prompt=[],
    kl_prompt="",
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    # note that rome does not support batch editing.
    request = requests[0]

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    deltas = execute_rome(model, tok, request, hparams, system_prompt,random_context_prompt,kl_prompt)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

            # 释放临时变量
            del upd_matrix
            torch.cuda.empty_cache()

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    del deltas
    torch.cuda.empty_cache()
    gc.collect()

    return model, weights_copy

def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    system_prompt:str,
    random_context_prompt:List[str],
    kl_prompt:str
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params, hparams.model_name,random_context_prompt),
            system_prompt
        )
        print("Left vector shape:", left_vector.shape)

        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params, hparams.model_name,random_context_prompt),
            system_prompt,
            kl_prompt,
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

            del upd_matrix
            del left_vector
            del right_vector
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    del weights
    del weights_copy
    torch.cuda.empty_cache()
    gc.collect()
    
    return deltas

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )

def get_context_templates(model, tok, length_params, model_name, random_context_prompt):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        if model_name in  ["Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-7B-Instruct"]:
            chat_temp = True
        else:
            chat_temp = False

        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        random_context_prompt,
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                        chat_temp = chat_temp
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE