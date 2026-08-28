# from huggingface_hub import login
import sys
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ft_hparams import FTHyperParams
from rome_hparams import ROMEHyperParams
from memit_hparams import MEMITHyperParams
from alphaedit_hparams import AlphaEditHyperParams
from unke_hparams import unkeHyperParams
from wise_hparams import WISEHyperParams
from grace_hparams import GraceHyperParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import PROMPT_DICT
from utils import read_json_file,LANGS, write_json_file
import random
import numpy as np
import torch
from util_editing import nethook
from tqdm import tqdm
import argparse
from editors.FT import apply_ft_to_model
from editors.ROME.rome import apply_rome_to_model
from editors.MEMIT.memit import apply_memit_to_model
from editors.ALPHAEDIT.alphaedit import apply_AlphaEdit_to_model
from editors.UNKE.unke import apply_unke_to_model
from editors.WISE.wise_main import apply_wise_to_model
from editors.GRACE.grace_main import apply_grace_to_model
from mmlu_pred_eval import eval_mmlu
from datasets import load_dataset
from util_editing.generate import input_format
import pickle
from time import time

TLANGS = ['English','Chinese','German','French','Japanese','Arabic']

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_response(model, tok, requests, question_type, hparams, all_tlangs=[]):
    tok.padding_side = 'left'

    def qwen_query_batch(system_prompts, user_prompts, max_new_tokens=200, num_beams=5, do_sample=False, temperature=None, top_p=None, top_k=None):
        # 准备批量消息
        all_messages = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        
        # 批量应用模板
        inputs = [
            tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in all_messages
        ]
        
        # 批量编码
        encoded_inputs = tok(inputs, padding=True, return_tensors="pt").to(f'cuda:{hparams.device}')
        
        # 批量生成
        generated_ids = model.generate(
            **encoded_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # 提取生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(encoded_inputs.input_ids, generated_ids)
        ]
        
        # 解码响应
        responses = tok.batch_decode(generated_ids, skip_special_tokens=True)
    
        return responses
    
    responses= {}
    if question_type=='mmlu':
        for l in TLANGS:
            mmlu_input=[]
            for request in requests:
                mmlu_input.extend(request['mmlu'][l])
            cors, preds = eval_mmlu(model, tok, mmlu_input, hparams.model_name)
            responses[l]={'score': cors,'pred':preds}

    else:
        for l in all_tlangs:
            if hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
                system_prompts=[]
                user_prompts=[]
                for request in requests:
                    for i in range(len(request[question_type][l])):
                        system_prompts.append(request['sys_prompts_all'][l])
                        user_prompts.append(request[question_type][l][i])

                batch_size = 100
                responses[l]=[]
                for i in tqdm(range(0, len(user_prompts), batch_size)):
                    batch_sys_prompts = system_prompts[i:i+batch_size]
                    batch_user_prompts = user_prompts[i:i+batch_size]
                    
                    batch_responses = qwen_query_batch(
                        batch_sys_prompts,
                        batch_user_prompts
                    )
                    responses[l].extend(batch_responses)

    if hparams.alg_name in ['ROME', 'MEMIT', 'AlphaEdit']:
        tok.padding_side = 'right'

    return responses

def prepare_requests_qwen(editing_samples, generality_dict, locality_dict,slang, tlang, all_tlangs,idiom_span,mmlu,flores):
    requests=[]
    for ind, item in enumerate(editing_samples):
        id=item['id']
        sys_prompt = PROMPT_DICT['GENERAL_TRANS']['sys'].format(SLang = slang, TLang = tlang)
        prompt = PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = slang, Source=item['src'], TLang=tlang)
        target_new = item['tgt'][tlang]
        subject=idiom_span[str(id)]

        reliability = {}
        generality={}
        locality={}
        for l in all_tlangs:
            reliability[l] = [PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = slang, Source=item['src'], TLang=l)]
            generality[l] = [ PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = slang, Source=generality_dict[id]['generality_samples'][i]['src'], TLang=l) for i in range(3)]
            locality[l] = [ PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = slang, Source=locality_dict[id]['locality_samples'][i]['src'], TLang=l) for i in range(3)]
        sys_prompts_all ={}
        for l in all_tlangs: 
            sys_prompts_all[l] = PROMPT_DICT['GENERAL_TRANS']['sys'].format(SLang = slang, TLang = l)

        mmlu_sample={}
        for l in TLANGS:
            mmlu_sample[l]=mmlu[l][:100]

        flores_sample={}
        for l in all_tlangs:
            flores_sample[l]=[PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = slang, Source=ss, TLang=l) for ss in flores[slang][:100]]

        requests.append({
            'sys_prompt':sys_prompt,
            'prompt':prompt,
            'target_new': target_new,
            'subject':subject,
            'sys_prompts_all':sys_prompts_all,
            'reliability':reliability,
            'generality':generality,
            'locality':locality,
            'mmlu':mmlu_sample,
            'flores': flores_sample
        })
    return requests

def list2dict(listA):
    dictA = {}
    for item in listA:
        id = item['id']
        dictA[id] = item
    return dictA

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--editing_method", type=str, required=True, help="FT,ROME,MEMIT,AlphaEdit")
    parser.add_argument("--hparam_file", type=str, required=True, help="hyper-parameters file")
    parser.add_argument("--slang", type=str, required=True, help="source lang, en,zh")
    parser.add_argument("--tlang", type=str, required=True, help="target lang, en,zh,ja,fr,de,ar")
    parser.add_argument("--seed", type=int, required=True, help="random seed")

    args = parser.parse_args()

    set_seed(42)

    test_sequence_steps = [1,4,16,64,256,1024]
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
        apply_algo = apply_ft_to_model
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
        apply_algo = apply_rome_to_model
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
        apply_algo = apply_memit_to_model
    elif  args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
        apply_algo = apply_AlphaEdit_to_model
    elif args.editing_method == 'UNKE':
        editing_hparams = unkeHyperParams
        apply_algo = apply_unke_to_model
    elif  args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
        apply_algo = apply_wise_to_model
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
        apply_algo = apply_grace_to_model
    else:
        raise NotImplementedError

    hparams = editing_hparams.from_hparams(args.hparam_file)
    model_suffix = hparams.model_name.split("/")[-1].replace('.','-').replace('_','-')
    
    old_state = random.getstate()
    random.seed(args.seed)
    editing_samples = read_json_file(f"MEMT/Editing_Samples/{model_suffix}/{args.slang}2x.json")
    editing_samples = random.sample(editing_samples, max(test_sequence_steps))
    # random.shuffle(editing_samples)
    ids= [sam['id'] for sam in editing_samples]
    save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/{args.slang}_{args.tlang}-seed{args.seed}.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    write_json_file(ids, save_file)
    random.setstate(old_state)

    generality_samples = read_json_file(f"MEMT/ParaIdiomSent_generality/{args.slang}2x.json")
    locality_samples = read_json_file(f"MEMT/ParaIdiomSent_locality/{args.slang}2x.json")

    idiom_span = read_json_file(f"MEMT/idiom_span_{args.slang}2x.json")

    generality_dict = list2dict(generality_samples)
    locality_dict =  list2dict(locality_samples)

    mmlu=read_json_file('MEMT/locality_MMMLU_shuffle1000.json')
    flores=read_json_file('MEMT/locality_flores_shuffle1000.json')
    for key in mmlu.keys():
        mmlu[key]=mmlu[key]
    for key in flores:
        flores[key]=flores[key]
        
    all_tlangs = [l for l in TLANGS if l != LANGS[args.slang]]
    requests = prepare_requests_qwen(editing_samples, generality_dict, locality_dict, LANGS[args.slang], LANGS[args.tlang], all_tlangs, idiom_span, mmlu,flores)

    model = AutoModelForCausalLM.from_pretrained(hparams.model_name)
    model.to(f'cuda:{hparams.device}')
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    if hparams.alg_name in ['ROME', 'MEMIT', 'AlphaEdit']:
        tok.padding_side = 'right'

    
    system_prompt=PROMPT_DICT['GENERAL_TRANS']['sys'].format(SLang = LANGS[args.slang], TLang = LANGS[args.tlang])
    random_context_prompt = [f"Produce a random {LANGS[args.slang]} sequence of tokens.", f"Surprise me with a random mix of everyday {LANGS[args.slang]} words.", f"Please generate a completely random {LANGS[args.slang]} sentence.", f"Return a syntactically valid but randomly generated {LANGS[args.slang]} sentence.", f"Output a random combination of meaningful {LANGS[args.slang]} words."]
    kl_prompt = PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = LANGS[args.slang], Source='{}', TLang=LANGS[args.tlang])

    if hparams.alg_name in ['UNKE','WISE']:
        if args.tlang == 'ja' and args.slang=='zh':
            dataset = load_dataset("larryvrh/CCMatrix-v1-Ja_Zh-filtered", split="train")
            small_dataset = list(dataset.select(range(10000)))
            del dataset
            ex_datas_source=[ PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = LANGS[args.slang], Source=item[args.slang], TLang=LANGS[args.tlang]) for item in small_dataset]
            ex_datas_source_format  = input_format(tok, ex_datas_source, system_prompt=system_prompt,  chat_temp = True if hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"] else False)
            ex_datas = [ex_datas_source_format[i]+ small_dataset[i][args.tlang]  for i in range(len(ex_datas_source_format))]
        else:
            try:
                dataset = load_dataset("Helsinki-NLP/opus-100", f"{args.slang}-{args.tlang}",split="train")
            except:
                dataset = load_dataset("Helsinki-NLP/opus-100", f"{args.tlang}-{args.slang}",split="train")
            small_dataset = list(dataset.select(range(10000)))
            del dataset
            ex_datas_source=[ PROMPT_DICT['GENERAL_TRANS']['user'].format(SLang = LANGS[args.slang], Source=item['translation'][args.slang], TLang=LANGS[args.tlang]) for item in small_dataset]
            ex_datas_source_format  = input_format(tok, ex_datas_source, system_prompt=system_prompt,  chat_temp = True if hparams.model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"] else False)
            ex_datas = [ex_datas_source_format[i]+ small_dataset[i]['translation'][args.tlang]  for i in range(len(ex_datas_source_format))]
            
        if hparams.alg_name in ['WISE']:
            for i, request in  enumerate(requests):
                request.update(
                {
                    'loc_prompt': ex_datas[i]
                }
            )
        
    # time_list=[]
    for i, request in enumerate(tqdm(requests, total=len(requests))):
        random_elements=[]
        if hparams.alg_name in ['UNKE']:
            random_elements = random.sample(ex_datas, 20)
        # start = time()        
        edited_model, weights_copy = apply_algo(model, tok, [request], hparams, copy=False,return_orig_weights=True, system_prompt=system_prompt, random_context_prompt=random_context_prompt, kl_prompt=kl_prompt,cache_accu=True,ex_data=random_elements) 
        ## set copy=False to save memory
        # exec_time = time() - start
        # time_list.append(exec_time)

        step = i+1
        if step in test_sequence_steps:
            print(f"step: {step}")
    
            print("evaluate: reliability")
            r_responses = generate_response(edited_model, tok, requests[:step], 'reliability', hparams, all_tlangs)
           
            print("evaluate: generality")
            g_responses = generate_response(edited_model, tok, requests[:step], 'generality', hparams, all_tlangs)
        
            print("evaluate: locality-memt")
            l_responses = generate_response(edited_model, tok, requests[:step], 'locality', hparams, all_tlangs)

            print("evaluate: locality-mmlu")
            m_responses_ans = generate_response(edited_model, tok, requests[:1],  'mmlu', hparams)
            m_eval = {}
            m_responses={}
            for l in TLANGS:
                m_responses[l]=m_responses_ans[l]['pred']
                m_eval[l]=m_responses_ans[l]['score']

            print("evaluate: locality-flores")
            f_responses = generate_response(edited_model, tok, requests[:1],  'flores', hparams,all_tlangs)
          
            layers = hparams.layers
            layers_str = "_".join(map(str, layers))
            save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/step{step}/{args.slang}_{args.tlang}_layer{layers_str}_r-seed{args.seed}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(r_responses, save_file)
            save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/step{step}/{args.slang}_{args.tlang}_layer{layers_str}_g-seed{args.seed}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(g_responses, save_file)
            save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/step{step}/{args.slang}_{args.tlang}_layer{layers_str}_l-seed{args.seed}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(l_responses, save_file)
            save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/step{step}/{args.slang}_{args.tlang}_layer{layers_str}_m-seed{args.seed}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(m_responses, save_file)
            save_file = f'editing_sequential_eval/{model_suffix}/{args.editing_method}/step{step}/{args.slang}_{args.tlang}/mmlu_layer{layers_str}-seed{args.seed}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(m_eval, save_file)
            save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/step{step}/{args.slang}_{args.tlang}_layer{layers_str}_f-seed{args.seed}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(f_responses, save_file)

    # save_file = f'editing_sequential_results/{model_suffix}/{args.editing_method}/time-{args.slang}_{args.tlang}-seed{args.seed}.json'
    # os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # write_json_file(time_list, save_file)

if __name__ == "__main__":
    main()
