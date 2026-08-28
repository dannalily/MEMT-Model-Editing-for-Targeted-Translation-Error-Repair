"""Generate keep samples (good translations) from evaluation results."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import read_json_file, write_json_file, LANGS


# Configuration
CONFIG = {
    'metrics': ['comet', 'bleurt', 'metricx', 'claudeScale', 'claudeBi'],
    'translators': ['m2m100-1-2B', 'nllb-200-3-3B', 'Qwen2-5-3B-Instruct', 'Qwen2-5-7B-Instruct'],
    'thresholds': {
        'comet': 0.9,
        'bleurt': 0.8,
        'metricx': -1.5,
        'claudeScale': 80
    },
    'source_langs': ['en', 'zh'],
    'target_langs': ['en', 'zh', 'ja', 'de', 'fr', 'ar'],
    'base_paths': {
        'para_data': 'bench_Construction/MEMT/ParaIdiomSent',
        'translations': 'bench_Construction/MEMT/ParaIdiomSent_trans',
        'evaluations': 'bench_Construction/MEMT/ParaIdiomSent_eval',
        'keep_samples': 'bench_Construction/MEMT/Keep_Samples'
    }
}


def collect_bad_indices_for_metric(metric: str, result: Dict, threshold: float) -> List[int]:
    """
    Collect bad translation indices for a specific metric.
    
    Args:
        metric: Metric name
        result: Evaluation results dictionary
        threshold: Threshold value for determining bad translations
    
    Returns:
        List of bad indices
    """
    bad_indices = []
    
    if metric == 'claudeBi':
        scores = result['claudeBi']
        bad_indices.extend([i for i, sc in enumerate(scores) if sc == 0])
        
    elif metric == 'metricx':
        goods = [-sc for sc in result['metricx_good']]
        bads = [-sc for sc in result['metricx_bad']]
        
        bad_indices.extend([i for i, sc in enumerate(goods) if sc < threshold])
        bad_indices.extend([i for i, (g, b) in enumerate(zip(goods, bads)) if b >= g])
        
    else:
        goods = result[f'{metric}_good']
        bads = result[f'{metric}_bad']
        
        bad_indices.extend([i for i, sc in enumerate(goods) if sc < threshold])
        bad_indices.extend([i for i, (g, b) in enumerate(zip(goods, bads)) if b >= g])
    
    return bad_indices


def load_metric_scores(translator: str, source_lang: str, target_lang: str) -> Dict:
    """Load all metric scores for a language pair."""
    eval_root = f"{CONFIG['base_paths']['evaluations']}/{translator}"
    scores_summary = {}
    
    for metric in CONFIG['metrics']:
        file_path = f"{eval_root}/{metric}/{source_lang}_{target_lang}.json"
        result = read_json_file(file_path)
        
        if metric == 'claudeBi':
            scores_summary['claudeBi'] = result['claudeBi']
        else:
            scores_summary[f'{metric}_good'] = result[f'{metric}_good']
            scores_summary[f'{metric}_bad'] = result[f'{metric}_bad']
    
    return scores_summary


def find_bad_indices(translator: str, source_lang: str, target_lang: str) -> List[int]:
    """Find all bad translation indices for a language pair."""
    scores_summary = load_metric_scores(translator, source_lang, target_lang)
    bad_indices = []
    
    for metric in CONFIG['metrics']:
        if metric == 'claudeBi':
            result = {'claudeBi': scores_summary['claudeBi']}
            threshold = None
        else:
            result = {
                f'{metric}_good': scores_summary[f'{metric}_good'],
                f'{metric}_bad': scores_summary[f'{metric}_bad']
            }
            threshold = CONFIG['thresholds'][metric]
        
        bad_indices.extend(collect_bad_indices_for_metric(metric, result, threshold))
    
    return bad_indices


def create_keep_entry(index: int, para_data: List[Dict], translation: str,
                     target_lang: str, scores_summary: Dict) -> Dict:
    """Create a keep sample entry for a good translation."""
    return {
        "id": para_data[index]["id"],
        "idiom": para_data[index]["idiom"],
        "src": para_data[index]["src"],
        "tgt": {LANGS[target_lang]: para_data[index]["tgt"][LANGS[target_lang]]},
        "lit_tgt": {LANGS[target_lang]: para_data[index]["lit_tgt"][LANGS[target_lang]]},
        "correct_trans": {
            LANGS[target_lang]: {
                "trans": translation,
                "scores": {
                    "comet": {
                        "vs_tgt": scores_summary["comet_good"][index],
                        "vs_lit_tgt": scores_summary["comet_bad"][index]
                    },
                    "bleurt": {
                        "vs_tgt": scores_summary["bleurt_good"][index],
                        "vs_lit_tgt": scores_summary["bleurt_bad"][index]
                    },
                    "metricx": {
                        "vs_tgt": scores_summary["metricx_good"][index],
                        "vs_lit_tgt": scores_summary["metricx_bad"][index]
                    },
                    "claudeScale": {
                        "vs_tgt": scores_summary["claudeScale_good"][index],
                        "vs_lit_tgt": scores_summary["claudeScale_bad"][index]
                    },
                    "claudeBi": {
                        "vs_tgt": scores_summary["claudeBi"][index]
                    }
                }
            }
        }
    }


def process_language_pair(translator: str, source_lang: str, target_lang: str,
                          para_data: List[Dict], total_samples: int):
    """Process a single language pair and generate keep samples."""
    trans_file = f"{CONFIG['base_paths']['translations']}/{translator}/{source_lang}_{target_lang}.json"
    translations = read_json_file(trans_file)
    
    # Find bad indices
    bad_indices = find_bad_indices(translator, source_lang, target_lang)
    
    # Calculate keep indices (all indices minus bad ones)
    keep_indices = sorted(set(range(total_samples)) - set(bad_indices))
    
    # Load scores
    scores_summary = load_metric_scores(translator, source_lang, target_lang)
    
    # Create keep entries
    keep_entries = []
    for idx in keep_indices:
        entry = create_keep_entry(idx, para_data, translations[idx], target_lang, scores_summary)
        keep_entries.append(entry)
    
    # Save to file
    output_dir = Path(CONFIG['base_paths']['keep_samples']) / translator
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{source_lang}_{target_lang}.json"
    write_json_file(keep_entries, str(output_file))
    
    print(f"{translator}_{source_lang}_{target_lang} keep: {len(keep_indices)}")


def process_translator(translator: str, source_lang: str):
    """Process all language pairs for a translator and source language."""
    print(f"=== {translator} / source={source_lang} ===")
    
    # Load source data
    para_data_file = f"{CONFIG['base_paths']['para_data']}/{source_lang}2x.json"
    para_data = read_json_file(para_data_file)
    total_samples = len(para_data)
    
    for target_lang in CONFIG['target_langs']:
        if source_lang == target_lang:
            continue
        
        print(f"{translator}/{source_lang}_{target_lang}")
        process_language_pair(translator, source_lang, target_lang, para_data, total_samples)


def main():
    """Main execution."""
    for source_lang in CONFIG['source_langs']:
        for translator in CONFIG['translators']:
            process_translator(translator, source_lang)


if __name__ == "__main__":
    main()