"""Evaluate translation quality metrics across different models and language pairs."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import read_json_file


# Configuration
CONFIG = {
    'metrics': ['comet', 'bleurt', 'metricx', 'claudeScale', 'claudeBi'],
    'translators': ['m2m100-1-2B', 'nllb-200-3-3B', 'Qwen2-5-3B-Instruct', 'Qwen2-5-7B-Instruct'],
    'source_langs': ['en', 'zh'],
    'target_langs': ['en', 'zh', 'ja', 'de', 'fr', 'ar'],
    'eval_base_dir': 'bench_construction/MEMT/ParaIdiomSent_eval'
}


def calculate_average_score(scores: list) -> float:
    """Calculate average score rounded to 4 decimals."""
    return round(sum(scores) / len(scores), 4)


def get_metric_key(metric: str) -> str:
    """Get the appropriate key for metric in result dictionary."""
    return metric if metric == 'claudeBi' else f"{metric}_good"


def print_metric_results(metric: str, scores: list):
    """Print metric evaluation results."""
    print(f"metric: {metric}")
    print(f"sample number: {len(scores)}")
    print(f"avg score: {calculate_average_score(scores)}")
    print("============")


def evaluate_translation_pair(translator: str, source_lang: str, target_lang: str):
    """Evaluate all metrics for a translation pair."""
    print(f"{translator}/{source_lang}_{target_lang}")
    
    for metric in CONFIG['metrics']:
        file_path = Path(CONFIG['eval_base_dir']) / translator / metric / f"{source_lang}_{target_lang}.json"
        result = read_json_file(str(file_path))
        
        metric_key = get_metric_key(metric)
        scores = result[metric_key]
        print_metric_results(metric, scores)


def main():
    """Main execution."""
    for source_lang in CONFIG['source_langs']:
        for target_lang in CONFIG['target_langs']:
            if source_lang != target_lang:
                for translator in CONFIG['translators']:
                    evaluate_translation_pair(translator, source_lang, target_lang)


if __name__ == "__main__":
    main()