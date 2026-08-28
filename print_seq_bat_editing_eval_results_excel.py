"""
Collect and analyze sequential/batch translation editing results.
"""

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd

from utils import read_json_file, LANGS


def list_to_dict(items: List[Dict]) -> Dict:
    """
    Convert list of items to dictionary keyed by id.
    
    Args:
        items: List of dictionaries with 'id' field
        
    Returns:
        Dictionary keyed by id
    """
    return {item['id']: item for item in items}


def load_editing_samples(
    translator: str,
    slang: str,
    tlang: str,
    step: int,
    seed: int,
    editing_type: str
) -> List[Dict]:
    """
    Load editing samples for the specified configuration.
    
    Args:
        translator: Model name
        slang: Source language
        tlang: Target language
        step: Number of editing steps
        seed: Random seed
        editing_type: Type of editing ('sequential' or 'batch')
        
    Returns:
        List of editing samples
    """
    # Load sample indices
    sample_indices = read_json_file(
        f'editing_{editing_type}_results/{translator}/FT/{slang}_{tlang}-seed{seed}.json'
    )
    
    # Load all samples and convert to dict
    all_samples = read_json_file(
        f"MEMT/Editing_Samples/{translator}/{slang}2x.json"
    )
    samples_dict = list_to_dict(all_samples)
    
    # Select samples based on indices
    selected_samples = [
        samples_dict[int(idx)] for idx in sample_indices[:step]
    ]
    
    return selected_samples


def load_pre_edit_scores(
    translator: str,
    metric: str,
    slang: str,
    test_type: str,
    editing_samples: List[Dict],
    all_tlangs: List[str],
    include_source: bool = False
) -> Tuple[Dict, Dict]:
    """
    Load scores before editing.
    
    Args:
        translator: Model name
        metric: Evaluation metric
        slang: Source language
        test_type: Type of test
        editing_samples: List of editing samples
        all_tlangs: Target languages (excluding source)
        include_source: Whether to include source language (for MMLU)
        
    Returns:
        Tuple of (before_editing_scores, results_dict)
    """
    before_editing = {}
    results = {}
    test_abbrev = test_type[:3]
    
    # Determine which languages to process
    langs = ([slang] + all_tlangs) if include_source else all_tlangs

    if test_type == 'reliability':
        re_language_score_dict = read_json_file(f'language_check_results/Editing_Samples_{translator}_{slang}2x.json')
    
        for tl in langs:
            scores = [
                item["wrong_trans"][LANGS[tl]]["scores"][metric]["vs_tgt"] if re_language_score_dict[str(item['id'])][tl] ==1 else 0
                for item in editing_samples
            ]
            before_editing[tl] = scores
            results[f'{tl}_{test_abbrev}_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_{test_abbrev}_Imp'] = 0.0

    elif test_type == 'generality':
        for tl in langs:
            eval_dict = read_json_file(
                f"MEMT/ParaIdiomSent_generality_eval/{translator}/{metric}/{slang}_{tl}.json"
            )
            ge_language_score_dict= read_json_file(f'language_check_results/generality_{translator}_{slang}2{tl}.json')
            scores = [
                eval_dict[str(item['id'])][i] if ge_language_score_dict[str(item['id'])][i] ==1 else 0
                for item in editing_samples
                for i in range(3)
            ]
            before_editing[tl] = scores
            results[f'{tl}_{test_abbrev}_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_{test_abbrev}_Imp'] = 0.0

    elif test_type == 'locality':
        for tl in langs:
            eval_dict = read_json_file(
                f"MEMT/ParaIdiomSent_locality_eval/{translator}/{metric}/{slang}_{tl}.json"
            )
            lo_language_score_dict= read_json_file(f'language_check_results/locality_{translator}_{slang}2{tl}.json')
            scores = [
                eval_dict[str(item['id'])][i] if lo_language_score_dict[str(item['id'])][i] ==1 else 0
                for item in editing_samples
                for i in range(3)
            ]
            before_editing[tl] = scores
            results[f'{tl}_{test_abbrev}_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_{test_abbrev}_Imp'] = 0.0

    elif test_type == 'flores':
        for tl in langs:
            scores_f = read_json_file(
                f"MEMT/locality_flores_eval/{translator}/{metric}/{slang}_{tl}.json"
            )[:100]
            flore_language_score = read_json_file(f'language_check_results/flores_{translator}_{slang}2{tl}.json')[:100]
            scores = [s if flore_language_score[i]==1 else 0 for i, s in enumerate(scores_f)]
            before_editing[tl] = scores
            results[f'{tl}_flo_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_flo_Imp'] = 0.0

    elif test_type == 'mmlu':
        mmlu_pre = read_json_file(
            f"MEMT/locality_MMMLU_eval/{translator}.json"
        )
        for tl in langs:
            scores = mmlu_pre[LANGS[tl]][:100]
            before_editing[tl] = scores
            results[f'{tl}_mml_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_mml_Imp'] = 0.0

    return before_editing, results


def compute_improvement(
    scores_after: List[float],
    scores_before: List[float],
    strict: bool = True
) -> float:
    """
    Calculate improvement ratio.
    
    Args:
        scores_after: Scores after editing
        scores_before: Scores before editing
        strict: If True, use >, otherwise use >=
        
    Returns:
        Improvement ratio
    """
    comparison = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
    improved = [
        1 if comparison(after, before) else 0
        for after, before in zip(scores_after, scores_before)
    ]
    return round(sum(improved) / len(improved), 4)


def load_post_edit_scores(
    translator: str,
    editing: str,
    slang: str,
    tlang: str,
    metric: str,
    suffix: str,
    test_type: str,
    before_editing: Dict,
    all_tlangs: List[str],
    step: int,
    seed: int,
    editing_type: str
) -> Dict:
    """
    Load and compute scores after editing.
    
    Args:
        translator: Model name
        editing: Editing method
        slang: Source language
        tlang: Target language
        metric: Evaluation metric
        suffix: Layer suffix
        test_type: Type of test
        before_editing: Pre-edit scores
        all_tlangs: Target languages (excluding source)
        step: Number of editing steps
        seed: Random seed
        editing_type: Type of editing ('sequential' or 'batch')
        
    Returns:
        Dictionary with results
    """
    results = {}
    test_abbrev = test_type[:3]
    step_name = {'sequential': 'step', 'batch': 'batch'}
    
    # Construct directory path
    dir_path = (
        f'editing_{editing_type}_eval/{translator}/{editing}/'
        f'{step_name[editing_type]}{step}/'
    )
    
    # Check if results exist
    if not os.path.exists(dir_path):
        # Return placeholder values if results don't exist
        langs = ([slang] + all_tlangs) if test_type == 'mmlu' else all_tlangs
        for tl in langs:
            results[f'{tl}_{test_abbrev}_score'] = -100
            results[f'{tl}_{test_abbrev}_Imp'] = -100
        return results
    
    # Load post-edit scores
    after_editing = read_json_file(
        f'{dir_path}{slang}_{tlang}/{metric}_{suffix}-seed{seed}.json'
    )
    mmlu_after = read_json_file(
        f'{dir_path}{slang}_{tlang}/mmlu_{suffix}-seed{seed}.json'
    )
    
    # Determine parameters based on test type
    if test_type in ['reliability', 'generality']:
        langs = all_tlangs
        strict = True
        language_dict=read_json_file(f'language_check_results/{editing_type}_edit/{translator}_{editing}/{step_name[editing_type]}{step}/{slang}_{tlang}/{test_type[0]}-seed{seed}.json')
    elif test_type in ['locality', 'flores']:
        langs = all_tlangs
        strict = False
        language_dict=read_json_file(f'language_check_results/{editing_type}_edit/{translator}_{editing}/{step_name[editing_type]}{step}/{slang}_{tlang}/{test_type[0]}-seed{seed}.json')
    elif test_type == 'mmlu':
        langs = [slang] + all_tlangs
        strict = False
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Compute scores and improvements
    for tl in langs:
        if test_type == 'mmlu':
            scores_after = mmlu_after[LANGS[tl]]
        else:
            scores_after = [s if language_dict[tl][i]==1 else 0 for i , s in enumerate(after_editing[test_type][LANGS[tl]])]

        scores_before = before_editing[tl]
        
        # Calculate average score
        results[f'{tl}_{test_abbrev}_score'] = round(
            sum(scores_after) / len(scores_after), 4
        )
        
        # Calculate improvement ratio
        results[f'{tl}_{test_abbrev}_Imp'] = compute_improvement(
            scores_after, scores_before, strict=strict
        )
    
    return results


def collect_results(
    translator: str,
    editing_list: List[str],
    slang: str,
    tlang: str,
    metric: str,
    suffix_list: List[str],
    test_type: str,
    all_tlangs: List[str],
    step: int,
    seed: int,
    editing_type: str
) -> pd.DataFrame:
    """
    Collect results across different editing methods.
    
    Args:
        translator: Model name
        editing_list: List of editing methods
        slang: Source language
        tlang: Target language
        metric: Evaluation metric
        suffix_list: Layer suffixes for each method
        test_type: Type of test
        all_tlangs: Target languages (excluding source)
        step: Number of editing steps
        seed: Random seed
        editing_type: Type of editing ('sequential' or 'batch')
        
    Returns:
        DataFrame with results
    """
    # Load editing samples
    editing_samples = load_editing_samples(
        translator, slang, tlang, step, seed, editing_type
    )
    
    # Pre-editing scores
    include_source = (test_type == 'mmlu')
    before_editing, pre_results = load_pre_edit_scores(
        translator, metric, slang, test_type, editing_samples,
        all_tlangs, include_source
    )
    all_results = {'pre-edit': pre_results}
    
    # Post-editing scores for each method
    for editing, suffix in zip(editing_list, suffix_list):
        results = load_post_edit_scores(
            translator, editing, slang, tlang, metric, suffix,
            test_type, before_editing, all_tlangs, step, seed, editing_type
        )
        all_results[editing] = results
    
    return pd.DataFrame.from_dict(all_results, orient='index')


def compute_average_results(
    translator: str,
    editing_list: List[str],
    slang: str,
    tlang: str,
    metric: str,
    suffix_list: List[str],
    test_type: str,
    all_tlangs: List[str],
    step: int,
    seeds: List[int],
    editing_type: str
) -> pd.DataFrame:
    """
    Compute average results across multiple seeds.
    
    Args:
        translator: Model name
        editing_list: List of editing methods
        slang: Source language
        tlang: Target language
        metric: Evaluation metric
        suffix_list: Layer suffixes for each method
        test_type: Type of test
        all_tlangs: Target languages (excluding source)
        step: Number of editing steps
        seeds: List of random seeds
        editing_type: Type of editing ('sequential' or 'batch')
        
    Returns:
        DataFrame with averaged results
    """
    all_seed_dfs = []
    
    # Collect results for all seeds
    for seed in seeds:
        try:
            df = collect_results(
                translator=translator,
                editing_list=editing_list,
                slang=slang,
                tlang=tlang,
                metric=metric,
                suffix_list=suffix_list,
                test_type=test_type,
                all_tlangs=all_tlangs,
                step=step,
                seed=seed,
                editing_type=editing_type
            )
            all_seed_dfs.append(df)
            print(f"  ✓ Processed step={step}, seed={seed}")
        except Exception as e:
            print(f"  ✗ Error processing step={step}, seed={seed}: {e}")
    
    # Calculate mean across seeds
    if all_seed_dfs:
        mean_df = pd.concat(all_seed_dfs).groupby(level=0).mean().round(4)
        return mean_df
    else:
        raise ValueError(f"No data collected for step={step}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect sequential/batch editing results and generate Excel files'
    )
    
    # Basic parameters
    parser.add_argument(
        '--translator',
        type=str,
        default='Qwen2-5-3B-Instruct',
        help='Translation model name'
    )
    parser.add_argument(
        '--slang',
        type=str,
        help='Source language code'
    )
    parser.add_argument(
        '--tlang',
        type=str,
        help='Target language code'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='bleurt',
        choices=['metricx', 'comet', 'bleurt', 'claudeScale', 'claudeBi'],
        help='Evaluation metric'
    )
    
    # Editing configuration
    parser.add_argument(
        '--editing-list',
        type=str,
        nargs='+',
        default=[],
        help='List of editing methods'
    )
    parser.add_argument(
        '--suffix-list',
        type=str,
        nargs='+',
        default=[],
        help='Layer suffixes for each editing method'
    )
    parser.add_argument(
        '--editing-type',
        type=str,
        choices=['sequential', 'batch'],
        help='Type of editing process'
    )
    
    # Test configuration
    parser.add_argument(
        '--test-types',
        type=str,
        nargs='+',
        default=['reliability', 'locality', 'generality', 'mmlu', 'flores'],
        help='Types of tests to evaluate'
    )
    
    # Language configuration
    parser.add_argument(
        '--target-langs',
        type=str,
        nargs='+',
        default=['en', 'zh', 'de', 'fr', 'ja', 'ar'],
        help='All target language codes'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 5, 10, 42, 123, 1234],
        help='List of random seeds for averaging'
    )
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[],
        help='List of editing steps to evaluate,[1,4,16,64,256,1024] or [1,4,16,32,64,128,256]'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for Excel files'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    if not args.suffix_list:
        if args.editing_type=='batch':
            if args.slang=='en':
                args.suffix_list=['layer7',  'layer5_6_7_8_9', 'layer5_6_7_8_9',
                    'layer7']
            else:
                args.suffix_list=['layer5','layer2_3_4_5_6', 'layer2_3_4_5_6',
                    'layer5']
        else:
            if args.slang=='en':
                args.suffix_list=['layer7', 'layer7', 'layer5_6_7_8_9', 'layer5_6_7_8_9',
                    'layer7', 'layer30', 'layer30']
            else:
                args.suffix_list=['layer5', 'layer5', 'layer2_3_4_5_6', 'layer2_3_4_5_6',
                    'layer5', 'layer30', 'layer30']
            
    if not args.editing_list:
        if args.editing_type=='batch':
            args.editing_list = ['FT', 'MEMIT', 'AlphaEdit', 'UNKE']
        else:
            args.editing_list = ['FT', 'ROME', 'MEMIT','AlphaEdit', 'UNKE', 'GRACE', 'WISE']

    if not args.steps:
        if args.editing_type=='batch':
            args.steps = [1,4,16,32,64,128,256]
        else:
            args.steps = [1,4,16,64,256,1024]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate inputs
    if len(args.editing_list) != len(args.suffix_list):
        raise ValueError(
            f"Number of editing methods ({len(args.editing_list)}) must match "
            f"number of suffixes ({len(args.suffix_list)})"
        )
    
    # Prepare language lists
    all_tlangs = [lang for lang in args.target_langs if lang != args.slang]
    out_pair_langs = [lang for lang in all_tlangs if lang != args.tlang]
    
    # Print configuration
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Translator:       {args.translator}")
    print(f"Language pair:    {args.slang} -> {args.tlang}")
    print(f"In-pair:          {args.slang}-{args.tlang}")
    print(f"Out-pair:         {args.slang}-{out_pair_langs} (average)")
    print(f"Metric:           {args.metric}")
    print(f"Editing type:     {args.editing_type}")
    print(f"Editing methods:  {', '.join(args.editing_list)}")
    print(f"Test types:       {', '.join(args.test_types)}")
    print(f"Steps:            {args.steps}")
    print(f"Seeds:            {args.seeds}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()
    
    # Process each test type and step
    for test_type in args.test_types:
        print(f"\n{'='*60}")
        print(f"Processing: {test_type}")
        print(f"{'='*60}\n")
        
        for step in args.steps:
            print(f"Step: {step}")
            print("-" * 60)
            
            try:
                # Compute averaged results across seeds
                mean_df = compute_average_results(
                    translator=args.translator,
                    editing_list=args.editing_list,
                    slang=args.slang,
                    tlang=args.tlang,
                    metric=args.metric,
                    suffix_list=args.suffix_list,
                    test_type=test_type,
                    all_tlangs=all_tlangs,
                    step=step,
                    seeds=args.seeds,
                    editing_type=args.editing_type
                )
                
                # Save results
                output_file = (
                    f'{args.output_dir}/{args.translator}/{args.slang}2{args.tlang}/{args.metric}_{test_type}_step{step}_avg.xlsx'
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                mean_df.to_excel(output_file)
                print(f"  ✓✓ Saved: {output_file}\n")
                
            except Exception as e:
                print(f"  ✗✗ Error for step={step}: {e}\n")
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()