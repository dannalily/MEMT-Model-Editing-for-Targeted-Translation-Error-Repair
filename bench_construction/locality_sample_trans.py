"""
Module for translating locality sample sentences using batch inference.

This script translates locality sample sentences from source languages
to target languages using Claude batch inference API, processing in batches
to handle large volumes of data.
"""

import logging
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    LANGS,
    read_json_file,
    read_jsonl_batch_inference,
    write_json_file
)
from models.claudeBatch_load_query import ClaudeBatchM
from prompt import PROMPT_DICT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================
MODEL_NAME = "us.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_TOKENS = 10000
TEMPERATURE = 0.7
BATCH_SIZE = 50000

SOURCE_LANGS = ['zh']
TARGET_LANGS = ['en', 'zh', 'ja', 'de', 'fr', 'ar']

S3_BUCKET = 'dann02'
S3_INPUT_DIR = 'locality_sample/input'
S3_OUTPUT_DIR = 'locality_sample/output'

INPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/locality_{lang}_sent.json'
OUTPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/locality_{slang}_{tlang}_trans.json'
BATCH_INPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/batch_cache/locality_{slang}_{tlang}_trans_batch{batch_num}.jsonl'
BATCH_OUTPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/batch_cache/locality_{slang}_{tlang}_trans_out_batch{batch_num}.jsonl'


# =========================
# Helper functions
# =========================
def prepare_prompts(
    data: List[List[dict]],
    target_lang: str
) -> Tuple[List[str], List[str]]:
    """
    Prepare system and user prompts for translation.
    
    Args:
        data: Nested list of sentence dictionaries
        target_lang: Target language code
        
    Returns:
        Tuple of (system_prompts, user_prompts) lists
    """
    system_prompts = []
    user_prompts = []
    target_lang_name = LANGS[target_lang]
    
    for item in data:
        for sentence in item:
            system_prompts.append(
                PROMPT_DICT['GENERAL_TRANS']['sys'].format(
                    SLang="English",
                    TLang=target_lang_name
                )
            )
            user_prompts.append(
                PROMPT_DICT['GENERAL_TRANS']['user'].format(
                    SLang="English",
                    Source=sentence['English'],
                    TLang=target_lang_name
                )
            )
    
    return system_prompts, user_prompts


def process_batch(
    system_prompts_batch: List[str],
    user_prompts_batch: List[str],
    source_lang: str,
    target_lang: str,
    batch_num: int
) -> List[str]:
    """
    Process a single batch of translations.
    
    Args:
        system_prompts_batch: Batch of system prompts
        user_prompts_batch: Batch of user prompts
        source_lang: Source language code
        target_lang: Target language code
        batch_num: Batch number identifier
        
    Returns:
        List of translated texts
    """
    try:
        logger.info(
            "Processing batch %d: %s -> %s (%d items)",
            batch_num,
            source_lang,
            target_lang,
            len(system_prompts_batch)
        )
        
        model = ClaudeBatchM(MODEL_NAME)
        
        # Prepare file paths
        local_input_file = BATCH_INPUT_PATH_TEMPLATE.format(
            slang=source_lang,
            tlang=target_lang,
            batch_num=batch_num
        )
        s3_input_file = f'{source_lang}_{target_lang}_batch{batch_num}.jsonl'
        local_output_file = BATCH_OUTPUT_PATH_TEMPLATE.format(
            slang=source_lang,
            tlang=target_lang,
            batch_num=batch_num
        )
        
        # Prepare input configuration
        model.prepare_input_config(
            system_prompts=system_prompts_batch,
            user_prompts=user_prompts_batch,
            local_input_file=local_input_file,
            s3_bucket=S3_BUCKET,
            s3_key=f'{S3_INPUT_DIR}/{s3_input_file}',
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        # Submit and wait for batch job
        job_arn = model.submit_and_wait(
            job_name=f'sentTrans-{source_lang}-{target_lang}-batch{batch_num}',
            s3_bucket=S3_BUCKET,
            s3_input_dir=S3_INPUT_DIR,
            s3_input_file=s3_input_file,
            s3_output_dir=S3_OUTPUT_DIR,
            local_output_file=local_output_file
        )
        
        logger.info("Batch %d completed: %s", batch_num, job_arn)
        
        # Read and return results
        results = read_jsonl_batch_inference(local_output_file, structured_out=False)
        translations = [item['pred'] for item in results]
        
        logger.info("Batch %d: %d translations extracted", batch_num, len(translations))
        
        return translations
        
    except Exception as e:
        logger.error(
            "Error processing batch %d (%s -> %s): %s",
            batch_num,
            source_lang,
            target_lang,
            str(e),
            exc_info=True
        )
        raise


def process_language_pair(
    data: List[List[dict]],
    source_lang: str,
    target_lang: str
) -> List[str]:
    """
    Process all translations for a language pair.
    
    Args:
        data: Nested list of sentence dictionaries
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        List of all translated texts
    """
    try:
        logger.info("Processing translation: %s -> %s", source_lang, target_lang)
        
        # Prepare all prompts
        logger.info("Preparing prompts...")
        system_prompts, user_prompts = prepare_prompts(data, target_lang)
        
        logger.info("Total prompts: %d", len(system_prompts))
        
        # Process in batches
        all_translations = []
        num_batches = (len(system_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info("Splitting into %d batches (batch size: %d)", num_batches, BATCH_SIZE)
        
        for batch_num, start_idx in enumerate(range(0, len(system_prompts), BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, len(system_prompts))
            
            system_prompts_batch = system_prompts[start_idx:end_idx]
            user_prompts_batch = user_prompts[start_idx:end_idx]
            
            batch_translations = process_batch(
                system_prompts_batch,
                user_prompts_batch,
                source_lang,
                target_lang,
                batch_num
            )
            
            all_translations.extend(batch_translations)
        
        # Write combined results
        output_path = OUTPUT_PATH_TEMPLATE.format(
            slang=source_lang,
            tlang=target_lang
        )
        write_json_file(all_translations, output_path)
        
        logger.info(
            "Completed %s -> %s: %d translations written to %s",
            source_lang,
            target_lang,
            len(all_translations),
            output_path
        )
        
        return all_translations
        
    except Exception as e:
        logger.error(
            "Error processing language pair %s -> %s: %s",
            source_lang,
            target_lang,
            str(e),
            exc_info=True
        )
        raise


def should_process_pair(source_lang: str, target_lang: str) -> bool:
    """
    Determine if a language pair should be processed.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        True if pair should be processed, False otherwise
    """
    # Skip if source and target are the same
    if source_lang == target_lang:
        return False
    
    # Skip zh -> en (already generated)
    if source_lang == 'zh' and target_lang == 'en':
        return False
    
    return True


def main() -> None:
    """Main function to process translations for all language pairs."""
    try:
        logger.info("Starting locality sentence translation")
        logger.info("Source languages: %s", SOURCE_LANGS)
        logger.info("Target languages: %s", TARGET_LANGS)
        
        for source_lang in SOURCE_LANGS:
            # Load source data
            input_path = INPUT_PATH_TEMPLATE.format(lang=source_lang)
            logger.info("Loading data from: %s", input_path)
            
            data = read_json_file(input_path)
            logger.info("Loaded %d items for %s", len(data), source_lang)
            
            # Process each target language
            for target_lang in TARGET_LANGS:
                if should_process_pair(source_lang, target_lang):
                    process_language_pair(data, source_lang, target_lang)
                else:
                    logger.info(
                        "Skipping %s -> %s (not needed)",
                        source_lang,
                        target_lang
                    )
        
        logger.info("All translation processing completed successfully")
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()