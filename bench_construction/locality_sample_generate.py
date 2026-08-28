"""
Module for generating locality sample sentences for idioms using batch inference.

This script processes idiom data and generates locality-focused sample sentences
using Claude batch inference API for English and Chinese idioms.
"""

import logging
import os
import sys
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
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
BATCH_SIZE = 10

SOURCE_LANGS = ['en', 'zh']
TARGET_LANGS = ['en', 'zh', 'ja', 'de', 'fr', 'ar']

S3_BUCKET = 'dann02'
S3_INPUT_DIR = 'locality_sample/input'
S3_OUTPUT_DIR = 'locality_sample/output'

IDIOM_KB_PATH_TEMPLATE = 'IdiomKB/{lang}_idiom_meaning_dedupe.json'
PARA_DATA_PATH_TEMPLATE = 'bench_construction/MEMT/ParaIdiomSent/{lang}2x.json'
BATCH_INPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/batch_cache/locality_{lang}_sent.jsonl'
BATCH_OUTPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/batch_cache/locality_{lang}_sent_out.jsonl'
FINAL_OUTPUT_PATH_TEMPLATE = 'bench_construction/intermediate_out/locality_{lang}_sent.json'

SCHEMA = {
    'zh': {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Chinese": {"type": "string"},
                        "English": {"type": "string"},
                    },
                    "required": ["Chinese", "English"],
                    "additionalProperties": False
                },
                "minItems": BATCH_SIZE,
                "maxItems": BATCH_SIZE
            }
        },
        "required": ["items"],
        "additionalProperties": False
    },
    'en': {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "English": {"type": "string"}
                    },
                    "required": ["English"],
                    "additionalProperties": False
                },
                "minItems": BATCH_SIZE,
                "maxItems": BATCH_SIZE
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }
}


# =========================
# Helper functions
# =========================
def build_id_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary mapping item IDs to items.
    
    Args:
        items: List of dictionaries containing 'id' key
        
    Returns:
        Dictionary mapping ID to item
    """
    return {item['id']: item for item in items}


def build_user_prompt(
    source_lang: str,
    idiom_data: Dict[str, Any]
) -> str:
    """
    Build user prompt for locality sentence generation.
    
    Args:
        source_lang: Source language code ('en' or 'zh')
        idiom_data: Idiom knowledge base item
        
    Returns:
        Formatted user prompt string
    """
    if source_lang == 'zh':
        return PROMPT_DICT["LOCALITY_ZH_EN"]["user"].format(
            Idiom=idiom_data['idiom'],
            ZH_Meaning=idiom_data['zh_meaning'],
            EN_Meaning=idiom_data['en_meaning']
        )
    else:  # en
        return PROMPT_DICT["LOCALITY_EN_EN"]["user"].format(
            Idiom=idiom_data['idiom'],
            EN_Meaning=idiom_data['en_meaning']
        )


def prepare_prompts(
    source_lang: str,
    idiom_kb: List[Dict[str, Any]],
    para_data: List[Dict[str, Any]]
) -> tuple[List[str], List[str]]:
    """
    Prepare system and user prompts for batch inference.
    
    Args:
        source_lang: Source language code ('en' or 'zh')
        idiom_kb: Idiom knowledge base data
        para_data: Parallel idiom sentence data
        
    Returns:
        Tuple of (system_prompts, user_prompts) lists
    """
    idiom_kb_map = build_id_map(idiom_kb)
    prompt_key = f"LOCALITY_{source_lang.upper()}_EN"
    
    system_prompts = [PROMPT_DICT[prompt_key]["sys"]] * len(para_data)
    user_prompts = []
    
    for item in para_data:
        item_id = item['id']
        idiom_data = idiom_kb_map[item_id]
        
        # Validate idiom consistency
        if item['idiom'] != idiom_data['idiom']:
            raise ValueError(
                f"Idiom mismatch for ID {item_id}: "
                f"{item['idiom']} != {idiom_data['idiom']}"
            )
        
        user_prompt = build_user_prompt(source_lang, idiom_data)
        user_prompts.append(user_prompt)
    
    return system_prompts, user_prompts


def process_language(source_lang: str) -> None:
    """
    Process locality sentence generation for a specific source language.
    
    Args:
        source_lang: Source language code ('en' or 'zh')
    """
    try:
        logger.info("Processing source language: %s", source_lang)
        
        # Load data
        idiom_kb_path = IDIOM_KB_PATH_TEMPLATE.format(lang=source_lang)
        para_data_path = PARA_DATA_PATH_TEMPLATE.format(lang=source_lang)
        
        logger.info("Loading idiom KB from: %s", idiom_kb_path)
        idiom_kb = read_json_file(idiom_kb_path)
        
        logger.info("Loading parallel data from: %s", para_data_path)
        para_data = read_json_file(para_data_path)
        
        logger.info("Loaded %d idioms, %d parallel items", len(idiom_kb), len(para_data))
        
        # Prepare prompts
        logger.info("Preparing prompts...")
        system_prompts, user_prompts = prepare_prompts(
            source_lang,
            idiom_kb,
            para_data
        )
        
        # Prepare batch inference
        logger.info("Preparing batch inference...")
        model = ClaudeBatchM(MODEL_NAME)
        
        local_input_file = BATCH_INPUT_PATH_TEMPLATE.format(lang=source_lang)
        s3_input_file = f'sentGen_{source_lang}.jsonl'
        
        model.prepare_input_config(
            system_prompts=system_prompts,
            user_prompts=user_prompts,
            local_input_file=local_input_file,
            s3_bucket=S3_BUCKET,
            s3_key=f'{S3_INPUT_DIR}/{s3_input_file}',
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            json_schema=SCHEMA[source_lang]
        )
        
        # Submit and wait for batch job
        logger.info("Submitting batch job...")
        local_output_file = BATCH_OUTPUT_PATH_TEMPLATE.format(lang=source_lang)
        
        job_arn = model.submit_and_wait(
            job_name=f'sentGen_{source_lang}',
            s3_bucket=S3_BUCKET,
            s3_input_dir=S3_INPUT_DIR,
            s3_input_file=s3_input_file,
            s3_output_dir=S3_OUTPUT_DIR,
            local_output_file=local_output_file
        )
        
        logger.info("Batch job completed: %s", job_arn)
        
        # Process results
        logger.info("Processing results...")
        results = read_jsonl_batch_inference(
            local_output_file,
            structured_out=True
        )
        
        samples = [item['pred']['items'] for item in results]
        
        output_path = FINAL_OUTPUT_PATH_TEMPLATE.format(lang=source_lang)
        write_json_file(samples, output_path)
        
        logger.info(
            "Completed %s: %d samples written to %s",
            source_lang,
            len(samples),
            output_path
        )
        
    except FileNotFoundError as e:
        logger.error("File not found for %s: %s", source_lang, str(e), exc_info=True)
        raise
    except KeyError as e:
        logger.error("Missing key for %s: %s", source_lang, str(e), exc_info=True)
        raise
    except Exception as e:
        logger.error(
            "Error processing language %s: %s",
            source_lang,
            str(e),
            exc_info=True
        )
        raise


def main() -> None:
    """Main function to process locality sentence generation for all languages."""
    try:
        logger.info("Starting locality sentence generation")
        logger.info("Source languages: %s", SOURCE_LANGS)
        
        for source_lang in SOURCE_LANGS:
            process_language(source_lang)
        
        logger.info("All language processing completed successfully")
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()