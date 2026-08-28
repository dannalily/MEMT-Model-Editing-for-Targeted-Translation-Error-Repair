import os
import argparse
import random
import numpy as np
import torch
from utils import *
from metrics import metricx23Scorer, cometScorer, bleurtScorer, claudeScorer

TLANGS = ['en','zh','de','fr','ja','ar']

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(metric, sources, refs, trans, slang=None, tlang=None,local_input_file=None,local_output_file=None, s3_bucket=None, s3_input_dir=None, s3_input_file=None, s3_output_dir=None, job_name=None,single_claude=False):

    # Load scorers
    if metric =='metricx':
        scorer = metricx23Scorer()
    elif metric == 'comet': 
        scorer = cometScorer()
    elif metric == 'bleurt':
        scorer = bleurtScorer()
    elif metric == 'claudeScale':
        scorer = claudeScorer('SCALE')
    elif metric ==  'claudeBi':
        scorer = claudeScorer('BINARY')
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric in ['metricx','bleurt']:
        scores = scorer.batch_score(trans, refs, batch_size=40)
        
    elif metric == 'comet':
        scores = scorer.batch_score(trans, refs, sources, batch_size=40)

    elif metric == 'claudeScale':
        if single_claude:
            scores = scorer.score_one_by_one(trans, refs, LANGS[slang], LANGS[tlang])
        else:
            scores = scorer.batch_score(trans, refs, LANGS[slang], LANGS[tlang],local_input_file,local_output_file, s3_bucket, s3_input_dir, s3_input_file, s3_output_dir, job_name)

    elif metric == 'claudeBi':
        if single_claude:
            scores = scorer.score_one_by_one(trans, refs, LANGS[slang], LANGS[tlang])
        else:
            scores = scorer.batch_score(trans, refs, LANGS[slang], LANGS[tlang],local_input_file,local_output_file, s3_bucket, s3_input_dir, s3_input_file, s3_output_dir, job_name)

    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return scores

def list2dict(listA):
    dictA = {}
    for item in listA:
        id = item['id']
        dictA[id] = item
    return dictA

def evaluate_all_metrics(args, editing_samples, generality_dict, locality_dict, flores, 
                         trans_r, trans_g, trans_l, trans_f):
    """Common evaluation logic for all types"""
    all_tlangs = [l for l in TLANGS if l != args.slang]
    
    print("eval reliability")
    scores_r = {}
    for tl in all_tlangs:
        l = LANGS[tl]
        sources_r = [item['src'] for item in editing_samples]
        refs_r_l = [item['tgt'][l] for item in editing_samples]
        trans_r_l = trans_r[l]

        if 'claude' in args.metric:
            if len(trans_r_l)<=100:
                scores_r[l] = evaluate(
                    args.metric, 
                    sources_r, 
                    refs_r_l, 
                    trans_r_l, 
                    args.slang,
                    tl,
                    single_claude=True)
            else:
                scores_r[l] = evaluate(
                    args.metric, 
                    sources_r, 
                    refs_r_l, 
                    trans_r_l, 
                    args.slang,
                    tl,
                    local_input_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_r.jsonl',
                    local_output_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_r_out.jsonl',
                    s3_bucket="dann02",
                    s3_input_dir="EditingEval/input",
                    s3_input_file=f'eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_r.jsonl',
                    s3_output_dir="EditingEval/output",
                    job_name=f"{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_r".replace('_','-'))
        else:
            scores_r[l] = evaluate(
                args.metric, 
                sources_r, 
                refs_r_l, 
                trans_r_l)

    print("eval generality")
    scores_g = {}
    for tl in all_tlangs:
        l = LANGS[tl]
        sources_g = [generality_dict[item['id']]['generality_samples'][i]['src'] for item in editing_samples for i in range(3)]
        refs_g_l = [generality_dict[item['id']]['generality_samples'][i]['tgt'][l] for item in editing_samples for i in range(3)]
        trans_g_l = trans_g[l]

        if 'claude' in args.metric:
            if len(trans_g_l)<=100:
                 scores_g[l] = evaluate(
                    args.metric, 
                    sources_g, 
                    refs_g_l, 
                    trans_g_l, 
                    args.slang,
                    tl,
                    single_claude=True)
            else:
                scores_g[l] = evaluate(
                    args.metric, 
                    sources_g, 
                    refs_g_l, 
                    trans_g_l, 
                    args.slang,
                    tl,
                    local_input_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_g.jsonl',
                    local_output_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_g_out.jsonl',
                    s3_bucket="dann02",
                    s3_input_dir="EditingEval/input",
                    s3_input_file=f'eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_g.jsonl',
                    s3_output_dir="EditingEval/output",
                    job_name=f"{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_g".replace('_','-'))
        else:
            scores_g[l] = evaluate(
                args.metric, 
                sources_g, 
                refs_g_l, 
                trans_g_l)

    print("eval locality")
    scores_l = {}
    for tl in all_tlangs:
        l = LANGS[tl]
        sources_l = [locality_dict[item['id']]['locality_samples'][i]['src'] for item in editing_samples for i in range(3)]
        refs_l_l = [locality_dict[item['id']]['locality_samples'][i]['tgt'][l] for item in editing_samples for i in range(3)]
        trans_l_l = trans_l[l]

        if 'claude' in args.metric:
            if len(trans_l_l)<=100:
                scores_l[l] = evaluate(
                    args.metric, 
                    sources_l, 
                    refs_l_l, 
                    trans_l_l, 
                    args.slang,
                    tl,
                    single_claude=True)
            else:
                scores_l[l] = evaluate(
                    args.metric, 
                    sources_l, 
                    refs_l_l, 
                    trans_l_l, 
                    args.slang,
                    tl,
                    local_input_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_l.jsonl',
                    local_output_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_l_out.jsonl',
                    s3_bucket="dann02",
                    s3_input_dir="EditingEval/input",
                    s3_input_file=f'eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_l.jsonl',
                    s3_output_dir="EditingEval/output",
                    job_name=f"{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_l".replace('_','-'))
        else:
            scores_l[l] = evaluate(
                args.metric, 
                sources_l, 
                refs_l_l, 
                trans_l_l)

    print("eval flores")
    scores_f = {}
    for tl in all_tlangs:
        l = LANGS[tl]
        sources_f = flores[LANGS[args.slang]]
        refs_f_l = flores[l]
        trans_f_l = trans_f[l]

        if 'claude' in args.metric:
            if len(trans_f_l)<=100:
                scores_f[l] = evaluate(
                    args.metric, 
                    sources_f, 
                    refs_f_l, 
                    trans_f_l, 
                    args.slang,
                    tl,
                    single_claude=True)
            else:
                scores_f[l] = evaluate(
                    args.metric, 
                    sources_f, 
                    refs_f_l, 
                    trans_f_l, 
                    args.slang,
                    tl,
                    local_input_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_f.jsonl',
                    local_output_file=f'editing_eval/intermediate_out/batch_cache/eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_f_out.jsonl',
                    s3_bucket="dann02",
                    s3_input_dir="EditingEval/input",
                    s3_input_file=f'eval_{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_f.jsonl',
                    s3_output_dir="EditingEval/output",
                    job_name=f"{args.type}_{args.metric}_{args.slang}_{args.tlang}_{l}_f".replace('_','-'))
        else:
            scores_f[l] = evaluate(
                args.metric, 
                sources_f, 
                refs_f_l, 
                trans_f_l)
    
    return {'reliability': scores_r, 'generality': scores_g, 'locality': scores_l, 'flores': scores_f}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True,
                        choices=["metricx", "comet", "bleurt", "claudeScale", "claudeBi"],
                        help="Which metric to run")
    parser.add_argument("--editing", type=str, required=True,
                        help="translation to be evaluated")
    parser.add_argument("--annotate", type=str, default='',
                        help="special anotate for cache file")
    parser.add_argument("--translator", type=str, required=True, 
                        help="")
    parser.add_argument("--slang", type=str, required=True,
                        help="source language, zh,en")
    parser.add_argument("--tlang", type=str, required=True,
                        help="target langauge, en,zh,ja,fr,de,ar")
    parser.add_argument("--type", type=str, required=True,
                        choices=["sequential", "batch", "single"],
                        help="sequential, batch, or single")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (required for sequential/batch)")

    args = parser.parse_args()

    # Validate seed requirement
    if args.type in ['sequential', 'batch'] and args.seed is None:
        parser.error(f"--seed is required for type '{args.type}'")
    
    # Set seed
    set_seed()

    # Load common data
    generality_samples = read_json_file(f"MEMT/ParaIdiomSent_generality/{args.slang}2x.json")
    locality_samples = read_json_file(f"MEMT/ParaIdiomSent_locality/{args.slang}2x.json")
    generality_dict = list2dict(generality_samples)
    locality_dict = list2dict(locality_samples)

    if args.type == 'single':
        # Single edit logic
        print(f"Running single edit evaluation...")
        max_sample_num = 1000
        editing_samples = read_json_file(f"MEMT/Editing_Samples/{args.translator}/{args.slang}2x.json")[:max_sample_num]
        flores = read_json_file('MEMT/locality_flores_shuffle1000.json')
        
        trans_r = read_json_file(f"editing_results/{args.translator}/{args.editing}/{args.slang}_{args.tlang}_{args.annotate}_r.json")
        trans_g = read_json_file(f"editing_results/{args.translator}/{args.editing}/{args.slang}_{args.tlang}_{args.annotate}_g.json")
        trans_l = read_json_file(f"editing_results/{args.translator}/{args.editing}/{args.slang}_{args.tlang}_{args.annotate}_l.json")
        trans_f = read_json_file(f"editing_results/{args.translator}/{args.editing}/{args.slang}_{args.tlang}_{args.annotate}_f.json")

        scores = evaluate_all_metrics(args, editing_samples, generality_dict, locality_dict, flores,
                                      trans_r, trans_g, trans_l, trans_f)

        out_file = f'editing_eval/{args.translator}/{args.editing}/{args.slang}_{args.tlang}/{args.metric}_{args.annotate}.json'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        write_json_file(scores, out_file)
        print(f"Results saved to {out_file}")

    else:
        # Sequential or batch logic
        print(f"Running {args.type} edit evaluation with seed {args.seed}...")
        sizes = {'sequential': [1, 4, 16, 64, 256, 1024], 'batch': [1, 4, 16, 32, 64, 128, 256]}
        steps = sizes[args.type]
        step_name = {'sequential': 'step', 'batch': 'batch'}

        editing_samples_ind = read_json_file(f'editing_{args.type}_results/{args.translator}/{args.editing}/{args.slang}_{args.tlang}-seed{args.seed}.json')
        editing_samples_all = read_json_file(f"MEMT/Editing_Samples/{args.translator}/{args.slang}2x.json")
        editing_samples_dict = list2dict(editing_samples_all)
        editing_samples_new = [editing_samples_dict[int(id_)] for id_ in editing_samples_ind]
        flores = read_json_file('MEMT/locality_flores_shuffle1000.json')

        for step in steps:
            print(f"\n{'='*50}")
            print(f"Processing {step_name[args.type]} {step}...")
            print(f"{'='*50}")
            
            editing_samples = editing_samples_new[:step]
            
            dir_path = f'editing_{args.type}_results/{args.translator}/{args.editing}/{step_name[args.type]}{step}/'
            if not os.path.exists(dir_path):
                continue
            trans_r = read_json_file(f'{dir_path}{args.slang}_{args.tlang}_{args.annotate}_r-seed{args.seed}.json')
            trans_g = read_json_file(f'{dir_path}{args.slang}_{args.tlang}_{args.annotate}_g-seed{args.seed}.json')
            trans_l = read_json_file(f'{dir_path}{args.slang}_{args.tlang}_{args.annotate}_l-seed{args.seed}.json')
            trans_f = read_json_file(f'{dir_path}{args.slang}_{args.tlang}_{args.annotate}_f-seed{args.seed}.json')

            # For flores, only use first 100 samples for sequential/batch
            flores_subset = {k: v[:100] for k, v in flores.items()}

            scores = evaluate_all_metrics(args, editing_samples, generality_dict, locality_dict, flores_subset,
                                          trans_r, trans_g, trans_l, trans_f)

            out_file = f'editing_{args.type}_eval/{args.translator}/{args.editing}/{step_name[args.type]}{step}/{args.slang}_{args.tlang}/{args.metric}_{args.annotate}-seed{args.seed}.json'
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            write_json_file(scores, out_file)
            print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()