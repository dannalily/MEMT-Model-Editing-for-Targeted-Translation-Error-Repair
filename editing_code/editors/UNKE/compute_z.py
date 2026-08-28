import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unke_hparams import unkeHyperParams
from util_editing import nethook
from util_editing.generate import input_format


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: unkeHyperParams,
    system_prompt:str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(data["target_new"], return_tensors="pt", add_special_tokens=False).to(f'cuda:{hparams.device}')[0]
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    input_tok = tok(
        input_format(tok,[data["prompt"]], system_prompt, chat_temp = True if hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"] else False),  
        return_tensors="pt",
        padding=True,
    ).to(f'cuda:{hparams.device}')
    
    input_ids = torch.cat([input_tok['input_ids'],torch.unsqueeze(target_ids[:-1], dim=0)],dim=1)
    del input_tok

    rewriting_targets = torch.tensor(-100, device=f'cuda:{hparams.device}').repeat(
        1, len(input_ids[0])
    )
   
    ex_len = len(input_ids[0])

    
    rewriting_targets[0, ex_len - len(target_ids) : ex_len] = target_ids


    lookup_idxs = [ex_len - len(target_ids)]
    

    
    loss_layer = max(hparams.v_loss_layer, layer)
    
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f'cuda:{hparams.device}')
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f'cuda:{hparams.device}')
    else:
        raise NotImplementedError
    target_init = None

    
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init  

        if cur_layer == hparams.layer_module_tmp.format(layer):
            
            if target_init is None:
               
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            
            for i, idx in enumerate(lookup_idxs):
                
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)  

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(input_ids).logits
            

        # Compute loss on rewriting targets
        output=tr[hparams.layer_module_tmp.format(loss_layer)].output
        del logits 
        torch.cuda.empty_cache()
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr =  output

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        del full_repr

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        del log_probs
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        del mask
        
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + weight_decay.to(nll_loss.device)
        print(
           f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}  + {np.round(weight_decay.item(), 3)} "
           f"avg prob of [{data['target_new']}] "
           f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
           break

        del nll_loss_each, weight_decay, nll_loss

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # 清理loss
        del loss

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
        
        # 清理trace对象
        del tr, output
        torch.cuda.empty_cache()

    target = target_init + delta  
    print(
       f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    del delta, target_init, opt
    del input_ids, rewriting_targets, target_ids, lookup_idxs
    del lm_w, lm_b, ln_f
    torch.cuda.empty_cache()

    return target



