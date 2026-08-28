"""Generate editing samples from translation evaluation results."""

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
        'comet': 0.7,
        'bleurt': 0.6,
        'metricx': -4,
        'claudeScale': 50
    },
    'source_langs': ['en', 'zh'],
    'target_langs': ['en', 'zh', 'ja', 'de', 'fr', 'ar'],
    'base_paths': {
        'para_data': 'bench_construction/MEMT/ParaIdiomSent',
        'translations': 'bench_construction/MEMT/ParaIdiomSent_trans',
        'evaluations': 'bench_construction/MEMT/ParaIdiomSent_eval',
        'editing_samples': 'bench_construction/MEMT/Editing_Samples'
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
        # Binary preference: 0 means bad
        scores = result['claudeBi']
        bad_indices.extend([i for i, sc in enumerate(scores) if sc == 0])
        
    elif metric == 'metricx':
        # MetricX scores are stored as negatives; invert for comparison
        goods = [-sc for sc in result['metricx_good']]
        bads = [-sc for sc in result['metricx_bad']]
        
        # Low vs_tgt score is bad
        bad_indices.extend([i for i, sc in enumerate(goods) if sc < threshold])
        # Literal better than reference is bad
        bad_indices.extend([i for i, (g, b) in enumerate(zip(goods, bads)) if b > g])
        
    else:
        # Common handling for comet, bleurt, claudeScale
        goods = result[f'{metric}_good']
        bads = result[f'{metric}_bad']
        
        # Low vs_tgt score is bad
        bad_indices.extend([i for i, sc in enumerate(goods) if sc < threshold])
        # Literal better than reference is bad
        bad_indices.extend([i for i, (g, b) in enumerate(zip(goods, bads)) if b > g])
    
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
    
    return sorted(set(bad_indices))


def create_entry(index: int, para_data: List[Dict], translation: str, 
                target_lang: str, scores_summary: Dict) -> Dict:
    """Create an editing sample entry for a bad translation."""
    return {
        "id": para_data[index]["id"],
        "idiom": para_data[index]["idiom"],
        "src": para_data[index]["src"],
        "tgt": {LANGS[target_lang]: para_data[index]["tgt"][LANGS[target_lang]]},
        "lit_tgt": {LANGS[target_lang]: para_data[index]["lit_tgt"][LANGS[target_lang]]},
        "wrong_trans": {
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


def update_aggregate_entry(aggregate_entries: Dict, index: int, para_data: List[Dict],
                           target_lang: str, wrong_trans_data: Dict):
    """Update or create aggregate entry with wrong translation for target language."""
    if index not in aggregate_entries:
        aggregate_entries[index] = {
            "id": para_data[index]["id"],
            "idiom": para_data[index]["idiom"],
            "src": para_data[index]["src"],
            "tgt": para_data[index]["tgt"],
            "lit_tgt": para_data[index]["lit_tgt"],
            "wrong_trans": {}
        }
    
    aggregate_entries[index]["wrong_trans"][LANGS[target_lang]] = wrong_trans_data


def process_language_pair(translator: str, source_lang: str, target_lang: str,
                          para_data: List[Dict]) -> tuple:
    """
    Process a single language pair and generate editing samples.
    
    Returns:
        Tuple of (bad_indices, editing_entries, scores_summary)
    """
    trans_file = f"{CONFIG['base_paths']['translations']}/{translator}/{source_lang}_{target_lang}.json"
    translations = read_json_file(trans_file)
    
    bad_indices = find_bad_indices(translator, source_lang, target_lang)
    scores_summary = load_metric_scores(translator, source_lang, target_lang)
    
    editing_entries = []
    for idx in bad_indices:
        entry = create_entry(idx, para_data, translations[idx], target_lang, scores_summary)
        editing_entries.append(entry)
    
    return bad_indices, editing_entries, scores_summary


def process_translator(translator: str, source_lang: str):
    """Process all language pairs for a translator and source language."""
    print(f"=== {translator} / source={source_lang} ===")
    
    # Load source data
    para_data_file = f"{CONFIG['base_paths']['para_data']}/{source_lang}2x.json"
    para_data = read_json_file(para_data_file)
    
    per_lang_bad_indices = []
    aggregate_entries = {}
    all_bad_indices = []
    
    for target_lang in CONFIG['target_langs']:
        if source_lang == target_lang:
            continue
        
        # Process this language pair
        bad_indices, editing_entries, scores_summary = process_language_pair(
            translator, source_lang, target_lang, para_data
        )
        
        per_lang_bad_indices.append(bad_indices)
        all_bad_indices.extend(bad_indices)
        
        # Update aggregate entries
        trans_file = f"{CONFIG['base_paths']['translations']}/{translator}/{source_lang}_{target_lang}.json"
        translations = read_json_file(trans_file)
        
        for idx in bad_indices:
            wrong_trans_data = editing_entries[bad_indices.index(idx)]["wrong_trans"][LANGS[target_lang]]
            update_aggregate_entry(aggregate_entries, idx, para_data, target_lang, wrong_trans_data)
        
        # Save per-language-pair file
        output_dir = Path(CONFIG['base_paths']['editing_samples']) / translator
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{source_lang}_{target_lang}.json"
        write_json_file(editing_entries, str(output_file))
        
        print(f"{translator}_{source_lang}_{target_lang} wrong: {len(bad_indices)}")
    
    # Find intersection of bad indices across all target languages
    bad_indices_common = sorted(set(per_lang_bad_indices[0]).intersection(*per_lang_bad_indices[1:]))
    editing_for_all = [aggregate_entries[idx] for idx in bad_indices_common]
    
    # Save cross-language file
    output_file = f"{CONFIG['base_paths']['editing_samples']}/{translator}/{source_lang}2x.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_json_file(editing_for_all, output_file)
    
    print(f"{translator}, all tlang wrong: {len(bad_indices_common)}")
    print(f"Saved {len(editing_for_all)} common bad cases to {output_file}")
    
    return list(set(all_bad_indices))


def main():
    """Main execution."""
    for source_lang in CONFIG['source_langs']:
        all_indices = []
        
        for translator in CONFIG['translators']:
            bad_indices = process_translator(translator, source_lang)
            all_indices.extend(bad_indices)
        
        unique_indices = sorted(set(all_indices))
        print(f"\n{source_lang}: {len(unique_indices)} unique bad samples")
        print(unique_indices)


if __name__ == "__main__":
    main()