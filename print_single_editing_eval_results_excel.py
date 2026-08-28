"""
Collect and analyze translation editing results.
"""

import argparse
from typing import Dict, List, Tuple

import pandas as pd
import os

from utils import read_json_file, LANGS


def load_pre_edit_scores(
    translator: str,
    slang: str,
    metric: str,
    test_type: str,
    all_tlangs: List[str],
    tlangs: List[str]
) -> Tuple[Dict, Dict]:
    """Load scores before editing."""
    editing_samples = read_json_file(
        f"MEMT/Editing_Samples/{translator}/{slang}2x.json"
    )[:1000]

    re_language_score_dict = read_json_file(f'language_check_results/Editing_Samples_{translator}_{slang}2x.json')
    
    before_editing = {}
    results = {}

    if test_type == 'reliability':
        for tl in all_tlangs:
            scores = [
                item["wrong_trans"][LANGS[tl]]["scores"][metric]["vs_tgt"] if re_language_score_dict[str(item['id'])][tl] ==1 else 0
                for item in editing_samples
            ]
            before_editing[tl] = scores
            results[f'{tl}_rel_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_rel_Imp'] = 0.0

    elif test_type == 'generality':
        for tl in all_tlangs:
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
            results[f'{tl}_gen_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_gen_Imp'] = 0.0

    elif test_type == 'locality':
        for tl in all_tlangs:
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
            results[f'{tl}_loc_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_loc_Imp'] = 0.0

    elif test_type == 'flores':
        for tl in all_tlangs:
            scores_f = read_json_file(
                f"MEMT/locality_flores_eval/{translator}/{metric}/{slang}_{tl}.json"
            )[:100]
            flore_language_score = read_json_file(f'language_check_results/flores_{translator}_{slang}2{tl}.json')[:100]
            scores = [s if flore_language_score[i]==1 else 0 for i, s in enumerate(scores_f)]
            before_editing[tl] = scores
            results[f'{tl}_flo_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_flo_Imp'] = 0.0

    elif test_type == 'mmlu':
        mmlu_pre = read_json_file(f"MEMT/locality_MMMLU_eval/{translator}.json")
        for tl in tlangs:
            scores = mmlu_pre[LANGS[tl]]
            before_editing[tl] = scores
            results[f'{tl}_mml_score'] = round(sum(scores) / len(scores), 4)
            results[f'{tl}_mml_Imp'] = 0.0

    return before_editing, results


def compute_improvement(
    scores_after: List[float],
    scores_before: List[float],
    strict: bool = True
) -> float:
    """Calculate improvement ratio."""
    comparison = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
    improved = [1 if comparison(after, before) else 0
                for after, before in zip(scores_after, scores_before)]
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
    tlangs: List[str]
) -> Dict:
    """Load and compute scores after editing."""
    after_editing = read_json_file(
        f"editing_eval/{translator}/{editing}/{slang}_{tlang}/{metric}_{suffix}.json"
    )
    mmlu_after = read_json_file(
        f"editing_eval/{translator}/{editing}/{slang}_{tlang}/mmlu_{suffix}.json"
    )

    results = {}
    test_abbrev = test_type[:3]

    if test_type in ['reliability', 'generality', 'locality', 'flores']:
        language_dict = read_json_file(f'language_check_results/single_edit/{translator}_{editing}/{slang}_{tlang}/{test_type[0]}.json')

        strict = test_type in ['reliability', 'generality']
        
        for tl in all_tlangs:
            scores_after = [s if language_dict[tl][i]==1 else 0 for i , s in enumerate(after_editing[test_type][LANGS[tl]])]
            scores_before = before_editing[tl]
            
            results[f'{tl}_{test_abbrev}_score'] = round(
                sum(scores_after) / len(scores_after), 4
            )
            results[f'{tl}_{test_abbrev}_Imp'] = compute_improvement(
                scores_after, scores_before, strict=strict
            )

    elif test_type == 'mmlu':
        for tl in tlangs:
            scores_after = mmlu_after[LANGS[tl]]
            scores_before = before_editing[tl]
            
            results[f'{tl}_mml_score'] = round(
                sum(scores_after) / len(scores_after), 4
            )
            results[f'{tl}_mml_Imp'] = compute_improvement(
                scores_after, scores_before, strict=False
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
    tlangs: List[str],
    all_tlangs: List[str]
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
        test_type: Type of test (reliability, generality, etc.)
        tlangs: All target languages
        all_tlangs: Target languages excluding source
    
    Returns:
        DataFrame with results
    """
    # Pre-editing scores
    before_editing, pre_results = load_pre_edit_scores(
        translator, slang, metric, test_type, all_tlangs, tlangs
    )
    all_results = {'pre-edit': pre_results}

    # Post-editing scores
    for editing, suffix in zip(editing_list, suffix_list):
        results = load_post_edit_scores(
            translator, editing, slang, tlang, metric, suffix,
            test_type, before_editing, all_tlangs, tlangs
        )
        all_results[editing] = results

    # Save results
    df = pd.DataFrame.from_dict(all_results, orient='index')
    output_file = f'excel_editing/results_{translator}/{slang}2{tlang}/{metric}_{test_type}.xlsx'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_excel(output_file)
    print(f"Saved: {output_file}")

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect translation editing results'
    )
    
    parser.add_argument(
        '--translator',
        type=str,
        default='Qwen2-5-3B-Instruct',
        help='Translation model name'
    )
    
    parser.add_argument(
        '--slang',
        type=str,
        default='en',
        help='Source language code'
    )
    
    parser.add_argument(
        '--tlang',
        type=str,
        default='zh',
        help='Target language code'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='bleurt',
        choices=["metricx", "comet", "bleurt", "claudeScale", "claudeBi"],
        help='Evaluation metric'
    )
    
    parser.add_argument(
        '--editing-methods',
        type=str,
        nargs='+',
        default=['FT', 'ROME', 'MEMIT', 'AlphaEdit', 'UNKE', 'GRACE', 'WISE'],
        help='Editing methods to evaluate'
    )
    
    parser.add_argument(
        '--suffixes',
        type=str,
        nargs='+',
        default=[],
        help='Layer suffixes for each method'
    )
    
    parser.add_argument(
        '--test-types',
        type=str,
        nargs='+',
        default=['reliability', 'locality', 'generality', 'mmlu', 'flores'],
        help='Test types to evaluate'
    )
    
    parser.add_argument(
        '--target-langs',
        type=str,
        nargs='+',
        default=['en', 'zh', 'de', 'fr', 'ja', 'ar'],
        help='All target language codes'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    if not args.suffixes:
        if args.slang=='en':
            args.suffixes=['layer7', 'layer7', 'layer5_6_7_8_9', 'layer5_6_7_8_9',
                 'layer7', 'layer30', 'layer30']
        else:
            args.suffixes=['layer5', 'layer5', 'layer2_3_4_5_6', 'layer2_3_4_5_6',
                 'layer5', 'layer30', 'layer30']
    
    # Validate inputs
    if len(args.editing_methods) != len(args.suffixes):
        raise ValueError("Number of editing methods must match number of suffixes")
    
    all_tlangs = [lang for lang in args.target_langs if lang != args.slang]
    
    # Process each test type
    for test_type in args.test_types:
        print(f"\n{'='*60}")
        print(f"Processing: {test_type}")
        print(f"{'='*60}\n")
        
        df = collect_results(
            translator=args.translator,
            editing_list=args.editing_methods,
            slang=args.slang,
            tlang=args.tlang,
            metric=args.metric,
            suffix_list=args.suffixes,
            test_type=test_type,
            tlangs=args.target_langs,
            all_tlangs=all_tlangs
        )
        
        print(df)
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()