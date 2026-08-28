"""
Translation Quality Assessment Script

This module evaluates translation quality across multiple language pairs
using MetricX-23 scorer for ParaIdiomSent datasets.
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import read_json_file
from metrics import metricx23Scorer


# Configuration
CONFIG = {
    'model_name': 'google/metricx-23-qe-xl-v2p0',
    'tokenizer_name': 'google/mt5-xl',
    'batch_size': 40,
    'random_seed': 42,
    'target_languages': {
        'en2x': ['Chinese', 'German', 'French', 'Japanese', 'Arabic'],
        'zh2x': ['German', 'French', 'Japanese', 'Arabic']
    },
    'dataset_paths': {
        'para_idiom': 'bench_construction/MEMT/ParaIdiomSent',
        'generality': 'bench_construction/MEMT/ParaIdiomSent_generality',
        'locality': 'bench_construction/MEMT/ParaIdiomSent_locality'
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_scorer(model_name: str, tokenizer_name: str, is_qe: bool = True):
    """
    Initialize MetricX-23 scorer.
    
    Args:
        model_name: Path or name of the model
        tokenizer_name: Path or name of the tokenizer
        is_qe: Whether to use quality estimation mode
        
    Returns:
        Initialized scorer object
        
    Raises:
        Exception: If scorer initialization fails
    """
    try:
        scorer = metricx23Scorer(
            model_name_or_path=model_name,
            tokenizer_name_or_path=tokenizer_name,
            is_qe=is_qe
        )
        return scorer
    except Exception as e:
        logger.error(f"Failed to initialize scorer: {e}", exc_info=True)
        raise


def compute_translation_scores(
    scorer,
    translations: List[str],
    sources: List[str],
    batch_size: int
) -> float:
    """
    Compute average translation quality scores.
    
    Args:
        scorer: MetricX scorer instance
        translations: List of translated texts
        sources: List of source texts
        batch_size: Batch size for scoring
        
    Returns:
        Average translation score
        
    Raises:
        Exception: If scoring fails
    """
    try:
        scores = scorer.batch_score(translations, sources, batch_size=batch_size)
        return sum(scores) / len(scores)
    except Exception as e:
        logger.error(f"Failed to compute scores: {e}", exc_info=True)
        raise


def evaluate_direct_translations(
    data: List[Dict[str, Any]],
    scorer,
    target_languages: List[str],
    source_key: str,
    batch_size: int
) -> Dict[str, float]:
    """
    Evaluate direct translations from source to target languages.
    
    Args:
        data: List of data items containing translations
        scorer: MetricX scorer instance
        target_languages: List of target language names
        source_key: Key to access source text in data
        batch_size: Batch size for scoring
        
    Returns:
        Dictionary mapping language pairs to average scores
    """
    results = {}
    sources = [item[source_key] for item in data]
    
    for target_lang in target_languages:
        try:
            translations = [item['tgt'][target_lang] for item in data]
            avg_score = compute_translation_scores(
                scorer, translations, sources, batch_size
            )
            lang_pair = f"{source_key}->{target_lang}"
            results[lang_pair] = avg_score
            logger.info(f"{lang_pair}: {avg_score:.4f}")
        except Exception as e:
            logger.error(
                f"Failed to evaluate {target_lang}: {e}",
                exc_info=True
            )
    
    return results


def evaluate_pivot_translations(
    data: List[Dict[str, Any]],
    scorer,
    target_languages: List[str],
    source_key: str,
    pivot_lang: str,
    batch_size: int
) -> Dict[str, float]:
    """
    Evaluate translations using pivot language (source -> pivot -> target).
    
    Args:
        data: List of data items containing translations
        scorer: MetricX scorer instance
        target_languages: List of target language names
        source_key: Key to access source text in data
        pivot_lang: Intermediate pivot language
        batch_size: Batch size for scoring
        
    Returns:
        Dictionary mapping language pairs to average scores
    """
    results = {}
    sources = [item[source_key] for item in data]
    pivot_translations = [item['tgt'][pivot_lang] for item in data]
    
    # First evaluate source -> pivot
    try:
        pivot_score = compute_translation_scores(
            scorer, pivot_translations, sources, batch_size
        )
        lang_pair = f"{source_key}->{pivot_lang}"
        results[lang_pair] = pivot_score
        logger.info(f"{lang_pair}: {pivot_score:.4f}")
    except Exception as e:
        logger.error(f"Failed to evaluate pivot {pivot_lang}: {e}", exc_info=True)
    
    # Then evaluate pivot -> targets
    for target_lang in target_languages:
        try:
            translations = [item['tgt'][target_lang] for item in data]
            avg_score = compute_translation_scores(
                scorer, translations, pivot_translations, batch_size
            )
            lang_pair = f"{pivot_lang}->{target_lang}"
            results[lang_pair] = avg_score
            logger.info(f"{lang_pair}: {avg_score:.4f}")
        except Exception as e:
            logger.error(
                f"Failed to evaluate {target_lang}: {e}",
                exc_info=True
            )
    
    return results


def check_para_idiom_sent(
    dataset_path: str,
    scorer,
    config: Dict[str, Any]
) -> None:
    """
    Check translation quality for ParaIdiomSent dataset.
    
    Args:
        dataset_path: Path to dataset directory
        scorer: MetricX scorer instance
        config: Configuration dictionary
    """
    logger.info(f"Checking {dataset_path}/en2x.json")
    try:
        data = read_json_file(f"{dataset_path}/en2x.json")
        evaluate_direct_translations(
            data, scorer, config['target_languages']['en2x'],
            'src_plain', config['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to process en2x.json: {e}", exc_info=True)
    
    logger.info(f"Checking {dataset_path}/zh2x.json")
    try:
        data = read_json_file(f"{dataset_path}/zh2x.json")
        evaluate_pivot_translations(
            data, scorer, config['target_languages']['zh2x'],
            'src_plain', 'English', config['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to process zh2x.json: {e}", exc_info=True)


def check_generality_locality(
    dataset_path: str,
    scorer,
    config: Dict[str, Any],
    sample_key: str,
    source_key: str
) -> None:
    """
    Check translation quality for generality/locality datasets.
    
    Args:
        dataset_path: Path to dataset directory
        scorer: MetricX scorer instance
        config: Configuration dictionary
        sample_key: Key to access samples ('generality_samples' or 'locality_samples')
        source_key: Key to access source text in samples
    """
    logger.info(f"Checking {dataset_path}/en2x.json")
    try:
        data = read_json_file(f"{dataset_path}/en2x.json")
        flat_data = [
            subitem for item in data for subitem in item[sample_key]
        ]
        evaluate_direct_translations(
            flat_data, scorer, config['target_languages']['en2x'],
            source_key, config['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to process en2x.json: {e}", exc_info=True)
    
    logger.info(f"Checking {dataset_path}/zh2x.json")
    try:
        data = read_json_file(f"{dataset_path}/zh2x.json")
        flat_data = [
            subitem for item in data for subitem in item[sample_key]
        ]
        evaluate_pivot_translations(
            flat_data, scorer, config['target_languages']['zh2x'],
            source_key, 'English', config['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to process zh2x.json: {e}", exc_info=True)


def main() -> None:
    """Main execution function."""
    try:
        # Initialize
        set_seed(CONFIG['random_seed'])
        scorer = initialize_scorer(
            CONFIG['model_name'],
            CONFIG['tokenizer_name']
        )
        
        # Check ParaIdiomSent
        check_para_idiom_sent(
            CONFIG['dataset_paths']['para_idiom'],
            scorer,
            CONFIG
        )
        
        # Check ParaIdiomSent_generality
        check_generality_locality(
            CONFIG['dataset_paths']['generality'],
            scorer,
            CONFIG,
            sample_key='generality_samples',
            source_key='src_plain'
        )
        
        # Check ParaIdiomSent_locality
        check_generality_locality(
            CONFIG['dataset_paths']['locality'],
            scorer,
            CONFIG,
            sample_key='locality_samples',
            source_key='src'
        )
        
        logger.info("Translation quality assessment completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()