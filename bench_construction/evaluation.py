"""Evaluate translation quality using various metrics."""

import sys
import os
import argparse
import random
from typing import Dict, List
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import read_json_file, write_json_file, LANGS
from metrics import metricx23Scorer, cometScorer, bleurtScorer, claudeScorer


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_scorer(metric: str):
    """Initialize the appropriate scorer based on metric."""
    scorers = {
        'metricx': lambda: metricx23Scorer(),
        'comet': lambda: cometScorer(),
        'bleurt': lambda: bleurtScorer(),
        'claudeScale': lambda: claudeScorer('SCALE'),
        'claudeBi': lambda: claudeScorer('BINARY')
    }
    
    if metric not in scorers:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return scorers[metric]()


def evaluate_metricx_bleurt(scorer, metric: str, translations: List[str], 
                            references: List[str], literals: List[str]) -> Dict:
    """Evaluate using MetricX or BLEURT (reference-based only)."""
    return {
        f'{metric}_bad': scorer.batch_score(translations, literals, batch_size=40),
        f'{metric}_good': scorer.batch_score(translations, references, batch_size=40)
    }


def evaluate_comet(scorer, translations: List[str], references: List[str], 
                   literals: List[str], sources: List[str]) -> Dict:
    """Evaluate using COMET (source + reference based)."""
    return {
        'comet_bad': scorer.batch_score(translations, literals, sources, batch_size=40),
        'comet_good': scorer.batch_score(translations, references, sources, batch_size=40)
    }


def evaluate_claude_scale(scorer, translations: List[str], references: List[str], 
                          literals: List[str], source_lang: str, target_lang: str,
                          claude_args: Dict) -> Dict:
    """Evaluate using Claude scale metric."""
    bad_input = claude_args['local_input_file'].replace('.jsonl', '_bad.jsonl')
    bad_output = claude_args['local_output_file'].replace('.jsonl', '_bad.jsonl')
    bad_s3_input = claude_args['s3_input_file'].replace('.jsonl', '_bad.jsonl')
    
    return {
        'claudeScale_bad': scorer.batch_score(
            translations, literals, source_lang, target_lang,
            bad_input, bad_output, claude_args['s3_bucket'], 
            claude_args['s3_input_dir'], bad_s3_input, 
            claude_args['s3_output_dir'], claude_args['job_name']
        ),
        'claudeScale_good': scorer.batch_score(
            translations, references, source_lang, target_lang,
            claude_args['local_input_file'], claude_args['local_output_file'],
            claude_args['s3_bucket'], claude_args['s3_input_dir'],
            claude_args['s3_input_file'], claude_args['s3_output_dir'],
            claude_args['job_name']
        )
    }


def evaluate_claude_binary(scorer, translations: List[str], references: List[str],
                           source_lang: str, target_lang: str, claude_args: Dict) -> Dict:
    """Evaluate using Claude binary metric."""
    return {
        'claudeBi': scorer.batch_score(
            translations, references, source_lang, target_lang,
            claude_args['local_input_file'], claude_args['local_output_file'],
            claude_args['s3_bucket'], claude_args['s3_input_dir'],
            claude_args['s3_input_file'], claude_args['s3_output_dir'],
            claude_args['job_name']
        )
    }


def evaluate(metric: str, sources: List[str], references: List[str], 
             literals: List[str], translations: List[str], 
             source_lang: str, target_lang: str, claude_args: Dict = None) -> Dict:
    """
    Evaluate translations using specified metric.
    
    Args:
        metric: Metric name
        sources: Source texts
        references: Reference translations
        literals: Literal translations
        translations: Translations to evaluate
        source_lang: Source language code
        target_lang: Target language code
        claude_args: Claude-specific arguments (only for claude metrics)
    
    Returns:
        Dictionary of scores
    """
    scorer = initialize_scorer(metric)
    
    if metric in ['metricx', 'bleurt']:
        return evaluate_metricx_bleurt(scorer, metric, translations, references, literals)
    
    elif metric == 'comet':
        return evaluate_comet(scorer, translations, references, literals, sources)
    
    elif metric == 'claudeScale':
        return evaluate_claude_scale(
            scorer, translations, references, literals,
            LANGS[source_lang], LANGS[target_lang], claude_args
        )
    
    elif metric == 'claudeBi':
        return evaluate_claude_binary(
            scorer, translations, references,
            LANGS[source_lang], LANGS[target_lang], claude_args
        )
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate translation quality")
    
    parser.add_argument("--metric", type=str, required=True,
                       choices=["metricx", "comet", "bleurt", "claudeScale", "claudeBi"],
                       help="Metric to use")
    parser.add_argument("--source_data", type=str, required=True,
                       help="Source data file")
    parser.add_argument("--trans_data", type=str, required=True,
                       help="Translation data file")
    parser.add_argument("--out_file", type=str, required=True,
                       help="Output file for results")
    parser.add_argument("--slang", type=str, required=True,
                       help="Source language (zh, en)")
    parser.add_argument("--tlang", type=str, required=True,
                       help="Target language (en, zh, ja, fr, de, ar)")
    
    # Claude-specific arguments
    parser.add_argument("--local_input_file", type=str, default='',
                       help="Local input file (Claude only)")
    parser.add_argument("--local_output_file", type=str, default='',
                       help="Local output file (Claude only)")
    parser.add_argument("--s3_bucket", type=str, default='',
                       help="S3 bucket (Claude only)")
    parser.add_argument("--s3_input_dir", type=str, default='',
                       help="S3 input directory (Claude only)")
    parser.add_argument("--s3_input_file", type=str, default='',
                       help="S3 input file (Claude only)")
    parser.add_argument("--s3_output_dir", type=str, default='',
                       help="S3 output directory (Claude only)")
    parser.add_argument("--job_name", type=str, default='',
                       help="Job name (Claude only)")
    
    return parser.parse_args()


def main():
    """Main execution."""
    set_seed()
    args = parse_arguments()
    
    # Load data
    source_data = read_json_file(args.source_data)
    trans_data = read_json_file(args.trans_data)
    
    # Extract fields
    sources = [item['src'] for item in source_data]
    references = [item['tgt'][LANGS[args.tlang]] for item in source_data]
    literals = [item['lit_tgt'][LANGS[args.tlang]] for item in source_data]
    translations = [item[f'{args.tlang}_trans'] for item in trans_data]
    
    # Prepare Claude arguments if needed
    claude_args = {
        'local_input_file': args.local_input_file,
        'local_output_file': args.local_output_file,
        's3_bucket': args.s3_bucket,
        's3_input_dir': args.s3_input_dir,
        's3_input_file': args.s3_input_file,
        's3_output_dir': args.s3_output_dir,
        'job_name': args.job_name
    } if args.metric.startswith('claude') else None
    
    # Evaluate and save
    scores = evaluate(
        args.metric, sources, references, literals, translations,
        args.slang, args.tlang, claude_args
    )
    
    # Ensure output directory exists
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    write_json_file(scores, args.out_file)
    
    print(f"Evaluation completed. Results saved to {args.out_file}")


if __name__ == "__main__":
    main()