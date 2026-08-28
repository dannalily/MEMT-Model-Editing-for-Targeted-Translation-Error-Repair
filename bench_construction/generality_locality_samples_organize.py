"""
Module for merging parallel idiom sentences with generality/locality samples.

This script combines parallel idiom sentence data with sample translations
into a unified format.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import LANGS, read_json_file, write_json_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
SOURCE_LANGS = ['en', 'zh']
TARGET_LANGS = ['en', 'zh', 'ja', 'de', 'fr', 'ar']
SAMPLE_TYPES = ['generality', 'locality']


def should_skip_pair(source_lang: str, target_lang: str) -> bool:
    """Check if language pair should be skipped."""
    return source_lang == target_lang or (source_lang == 'zh' and target_lang == 'en')


def process_sample_type(sample_type: str, source_lang: str) -> None:
    """Process and merge data for a specific sample type and source language."""
    try:
        logger.info("Processing: %s - %s", sample_type, source_lang)
        
        # Load data
        para_data = read_json_file(f'bench_construction/MEMT/ParaIdiomSent/{source_lang}2x.json')
        sample_data = read_json_file(f'bench_construction/intermediate_out/{sample_type}_{source_lang}_sent.json')
        
        # Load translations
        trans_data = {}
        for target_lang in TARGET_LANGS:
            if not should_skip_pair(source_lang, target_lang):
                trans_path = f'bench_construction/intermediate_out/{sample_type}_{source_lang}_{target_lang}_trans.json'
                trans_data[target_lang] = read_json_file(trans_path)
        
        logger.info("Loaded %d items", len(para_data))
        
        # Merge data
        merged_samples = []
        trans_index = 0
        
        for item_idx, sample_item in enumerate(sample_data):
            merged_item = {
                'id': para_data[item_idx]['id'],
                'idiom': para_data[item_idx]['idiom'],
            }
            
            subsamples = []
            for subitem in sample_item:
                # Build subsample
                subsample = {'src': subitem[LANGS[source_lang]]}
                
                # Add plain text for generality
                if sample_type == 'generality':
                    plain_key = 'Plain_Chinese' if source_lang == 'zh' else 'Plain_English'
                    subsample['src_plain'] = subitem[plain_key]
                
                # Add translations
                subsample['tgt'] = {'English': subitem['English']} if source_lang == 'zh' else {}
                
                for target_lang in TARGET_LANGS:
                    if not should_skip_pair(source_lang, target_lang):
                        subsample['tgt'][LANGS[target_lang]] = trans_data[target_lang][trans_index]
                
                subsamples.append(subsample)
                trans_index += 1
            
            merged_item[f'{sample_type}_samples'] = subsamples
            merged_samples.append(merged_item)
        
        # Write output
        output_path = f'bench_construction/MEMT/ParaIdiomSent_{sample_type}/{source_lang}2x.json'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_json_file(merged_samples, output_path)
        
        logger.info("Completed: %d items -> %s", len(merged_samples), output_path)
        
    except Exception as e:
        logger.error("Error processing %s-%s: %s", sample_type, source_lang, str(e), exc_info=True)
        raise


def main() -> None:
    """Main function to process all sample types and source languages."""
    try:
        logger.info("Starting merge: %s x %s", SAMPLE_TYPES, SOURCE_LANGS)
        
        for sample_type in SAMPLE_TYPES:
            for source_lang in SOURCE_LANGS:
                process_sample_type(sample_type, source_lang)
        
        logger.info("All merging completed successfully")
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()