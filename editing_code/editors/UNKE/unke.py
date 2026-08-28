import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(__file__))

from transformers import AutoModelForCausalLM, AutoTokenizer
from unke_hparams import unkeHyperParams
from util_editing import nethook
from editors.UNKE.compute_z import compute_z
import torch
import torch.nn as nn
from util_editing.generate import input_format
import torch.optim as optim
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
# import inspect
# print(inspect.getfile(compute_z))
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeHyperParams,
    layer: int,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(f'cuda:{hparams.device}')
    idxs = [i.sum()-1 for i in input_ids['attention_mask']]
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=False,
            ) as tr:
                _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list=[]
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i,idxs[i]])
    zs_out =torch.stack(zs_out_list,dim=0)


    return zs_out,idxs

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # and 'mlp' in n
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters

def apply_unke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data:list,
    hparams:unkeHyperParams,
    ex_data:list,
    system_prompt:str,
    **kwargs):

    preserve_params = []
    for name, params in model.named_parameters():
        #print(name)
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    z_layer = hparams.layers[-1]
    z_list = []
    for data in batch_data:
        
        cur_z = compute_z(   
            model,
            tok,
            data,
            z_layer,
            hparams,
            system_prompt
        )

        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=0)
    batch_question = input_format(tok,[i['prompt'] for i in batch_data], system_prompt, chat_temp = True if hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"] else False)

    for i, layer in enumerate(hparams.layers):
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=False,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        layer_out_ks =layer_out_ks.clone()

        cur_zs,idxs = compute_ks(model, tok,batch_question, hparams, z_layer)
        cur_zs=cur_zs.clone()
        
        targets = (zs - cur_zs).detach()
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=False,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out

        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers(1,4096)

        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))

        # get_qwen2_causal_mask
        if  hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
            input_causal_mask,input_position_ids = get_qwen2_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids = get_qwen2_causal_mask(stat_in,ex_tok['attention_mask'])

            with torch.no_grad():
                rotary = getattr(model,"model",model).rotary_emb
                in_cos, in_sin = rotary(layer_in_ks, input_position_ids)
                ex_cos, ex_sin = rotary(stat_in, ex_position_ids)
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,hparams.lr)
        
        
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        
        for i in range(len(idxs)):
            layer_out_ks[i,idxs[i]]+=resid[i].detach()
        
        for step in range(hparams.optim_num_step):
            #scheduler.step()
            optimizer.zero_grad()
            loss=None
            if hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
                loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids, position_embeddings=(ex_cos,ex_sin)), stat_out)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,position_embeddings=(in_cos,in_sin)), layer_out_ks)
     
            if loss < 5e-4:
                break
            
            loss.backward(retain_graph=False)
            optimizer.step()     
            print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, hparams.optim_num_step, loss.item(),layer))

        vars_to_delete = [
            layer_in_ks, layer_out_ks, stat_in, stat_out,
            input_causal_mask, input_position_ids, ex_causal_mask, ex_position_ids,
            in_cos, in_sin, ex_cos, ex_sin
        ]
        
        for var in vars_to_delete:
            if var is not None:
                del var
        
        del optimizer, params, _layer
        torch.cuda.empty_cache()
    del ex_tok, zs, batch_question
    torch.cuda.empty_cache()

    return model, weights_copy


def get_qwen2_causal_mask(input_tensor,attention_mask,past_key_values_length = 0):
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )

    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_tensor.shape[0], input_tensor.shape[1]),
            input_tensor,
            0,
        )

    return attention_mask,position_ids