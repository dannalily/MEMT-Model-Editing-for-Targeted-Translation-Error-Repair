"""
Module for generating parallel idiom sentences across multiple languages.

This script processes English and Chinese idioms, generates contextual sentences,
and translates them to multiple target languages (German, French, Chinese/English, Japanese, Arabic).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make parent importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.claude_load_query import ClaudeM
from utils import (
    write_json_file,
    run_multithreading,
    read_json_file,
    query_with_retries,
)
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
MAX_RETRIES = 8
TEMPERATURE = 0.7
MAX_TOKENS = 10000
THREADS = 10

TARGET_LANGS = ["German", "French", "Chinese", "Japanese", "Arabic", "English"]

INPUT_EN_PATH = "IdiomKB/en_idiom_meaning_dedupe.json"
INPUT_ZH_PATH = "IdiomKB/zh_idiom_meaning_dedupe.json"
OUTPUT_DIR = Path("bench_construction/MEMT/ParaIdiomSent")

SCHEMA_ZH_EN = {
    "type": "object",
    "properties": {
        "Chinese": {"type": "string"},
        "Plain_Chinese": {"type": "string"},
        "English": {"type": "string"},
    },
    "required": ["Chinese", "Plain_Chinese", "English"],
    "additionalProperties": False,
}

SCHEMA_EN_EN = {
    "type": "object",
    "properties": {
        "English": {"type": "string"},
        "Plain_English": {"type": "string"},
    },
    "required": ["English", "Plain_English"],
    "additionalProperties": False,
}


# =========================
# Input builders
# =========================
def inputs_en_idiom_en(items: List[dict]) -> List[Tuple[int, dict]]:
    """
    Build input payloads for English idiom sentence generation.
    
    Args:
        items: List of English idiom dictionaries
        
    Returns:
        Enumerated list of (index, payload) tuples
    """
    payloads: List[dict] = []
    for item in items:
        sys_prompt = PROMPT_DICT["SENT_IDIOM_EN_EN"]["sys"]
        user_prompt = PROMPT_DICT["SENT_IDIOM_EN_EN"]["user"].format(
            Idiom=item['idiom'],
            Meaning=item['en_meaning']
        )
        payloads.append({
            "id": item["id"],
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt
        })
    return list(enumerate(payloads))


def inputs_zh_idiom_en(items: List[dict]) -> List[Tuple[int, dict]]:
    """
    Build input payloads for Chinese idiom sentence generation.
    
    Args:
        items: List of Chinese idiom dictionaries
        
    Returns:
        Enumerated list of (index, payload) tuples
    """
    payloads: List[dict] = []
    for item in items:
        sys_prompt = PROMPT_DICT["SENT_IDIOM_ZH_EN"]["sys"]
        user_prompt = PROMPT_DICT["SENT_IDIOM_ZH_EN"]["user"].format(
            Idiom=item['idiom'],
            ZH_Meaning=item['zh_meaning'],
            EN_Meaning=item['en_meaning']
        )
        payloads.append({
            "id": item["id"],
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt
        })
    return list(enumerate(payloads))


def inputs_general_trans(
    items: List[dict],
    key: str,
    source_lang: str,
    target_lang: str
) -> List[Tuple[int, dict]]:
    """
    Build input payloads for general translation.
    
    Args:
        items: List of item dictionaries
        key: Key to extract source text from items
        source_lang: Source language name
        target_lang: Target language name
        
    Returns:
        Enumerated list of (index, payload) tuples
    """
    sys_prompt = PROMPT_DICT["GENERAL_TRANS"]["sys"].format(
        SLang=source_lang,
        TLang=target_lang
    )
    payloads: List[dict] = []
    for item in items:
        src_text = item.get(key, "")
        user_prompt = PROMPT_DICT["GENERAL_TRANS"]["user"].format(
            SLang=source_lang,
            Source=src_text,
            TLang=target_lang
        )
        payloads.append({
            "id": item["id"],
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "Tlang": target_lang
        })
    return list(enumerate(payloads))


def inputs_literal_trans(
    items: List[dict],
    key: str,
    source_lang: str,
    target_lang: str
) -> List[Tuple[int, dict]]:
    """
    Build input payloads for literal translation.
    
    Args:
        items: List of item dictionaries
        key: Key to extract source text from items
        source_lang: Source language name
        target_lang: Target language name
        
    Returns:
        Enumerated list of (index, payload) tuples
    """
    sys_prompt = PROMPT_DICT["LITERAL_TRANS"]["sys"].format(
        SLang=source_lang,
        TLang=target_lang
    )
    payloads: List[dict] = []
    for item in items:
        src_text = item.get(key, "")
        user_prompt = PROMPT_DICT["LITERAL_TRANS"]["user"].format(
            SLang=source_lang,
            Source=src_text,
            TLang=target_lang
        )
        payloads.append({
            "id": item["id"],
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "Tlang": target_lang
        })
    return list(enumerate(payloads))


# =========================
# Workers
# =========================
def worker_schema(item: Tuple[int, dict], schema: dict) -> Tuple[int, dict]:
    """
    Worker function for structured output generation with schema validation.
    
    Args:
        item: Tuple of (index, payload dictionary)
        schema: JSON schema for structured output
        
    Returns:
        Tuple of (index, result dictionary)
    """
    idx, payload = item
    try:
        gen = ClaudeM(MODEL_NAME)
        _, resp = query_with_retries(
            gen,
            payload["sys_prompt"],
            payload["user_prompt"],
            schema=schema,
            max_retries=MAX_RETRIES,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        structured = (resp or {}).get("structured_output") or {}
        return idx, {**{"id": payload["id"]}, **structured}
    except Exception as e:
        logger.error(
            "Worker schema failed for item %s: %s",
            payload.get("id", "unknown"),
            str(e),
            exc_info=True
        )
        raise


def worker_text(item: Tuple[int, dict]) -> Tuple[int, dict]:
    """
    Worker function for text generation without schema.
    
    Args:
        item: Tuple of (index, payload dictionary)
        
    Returns:
        Tuple of (index, result dictionary)
    """
    idx, payload = item
    try:
        gen = ClaudeM(MODEL_NAME)
        text, _ = query_with_retries(
            gen,
            payload["sys_prompt"],
            payload["user_prompt"],
            schema=None,
            max_retries=MAX_RETRIES,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return idx, {
            "id": payload["id"],
            payload.get("Tlang", "text"): (text or "")
        }
    except Exception as e:
        logger.error(
            "Worker text failed for item %s: %s",
            payload.get("id", "unknown"),
            str(e),
            exc_info=True
        )
        raise


# =========================
# Validation helpers
# =========================
def validate_item_fields(item: dict, required_fields: List[str], index: int) -> bool:
    """
    Validate that item contains all required non-empty fields.
    
    Args:
        item: Dictionary to validate
        required_fields: List of required field names
        index: Item index for logging
        
    Returns:
        True if valid, False otherwise
    """
    for field in required_fields:
        if not item.get(field):
            logger.warning("Missing or empty field '%s' at index %d", field, index)
            return False
    return True


# =========================
# Branch orchestrations
# =========================
def english_branch(out_dir: Path) -> List[dict]:
    """
    Process English idioms: generate sentences and translate to multiple languages.
    
    Args:
        out_dir: Output directory path
        
    Returns:
        List of processed idiom dictionaries
    """
    logger.info("Processing English idioms...")
    
    try:
        en_data = read_json_file(INPUT_EN_PATH)
        logger.info("Loaded %d English idioms", len(en_data))
        
        # Generate English sentences with idioms
        logger.info("Generating English sentences with idioms...")
        en_en_inputs = inputs_en_idiom_en(en_data)
        en_en_pairs = run_multithreading(
            lambda x: worker_schema(x, SCHEMA_EN_EN),
            en_en_inputs,
            pn=THREADS
        )
        
        # Literal translations
        literal_map: Dict[str, List[dict]] = {}
        for lang in TARGET_LANGS:
            if lang == "English":
                continue
            logger.info("Generating literal translation: English -> %s", lang)
            li = inputs_literal_trans(en_en_pairs, "English", "English", lang)
            literal_map[lang] = run_multithreading(worker_text, li, pn=THREADS)
        
        # General translations of plain text
        trans_map: Dict[str, List[dict]] = {}
        for lang in TARGET_LANGS:
            if lang == "English":
                continue
            logger.info("Translating plain English -> %s", lang)
            gi = inputs_general_trans(en_en_pairs, "Plain_English", "English", lang)
            trans_map[lang] = run_multithreading(worker_text, gi, pn=THREADS)
        
        # Merge results
        summary: List[dict] = []
        for ind, src in enumerate(en_data):
            item1 = en_en_pairs[ind]
            
            if not validate_item_fields(
                item1,
                ["id", "English", "Plain_English"],
                ind
            ) or item1["id"] != src["id"]:
                continue
            
            ans = {
                "id": src["id"],
                "idiom": src["idiom"],
                "src": item1["English"],
                "src_plain": item1["Plain_English"],
                "tgt": {},
                "lit_tgt": {},
            }
            
            is_valid = True
            for lang in TARGET_LANGS:
                if lang == "English":
                    continue
                
                literal_item = literal_map[lang][ind]
                trans_item = trans_map[lang][ind]
                
                if (
                    trans_item.get("id") != src["id"]
                    or literal_item.get("id") != src["id"]
                    or not trans_item.get(lang)
                    or not literal_item.get(lang)
                ):
                    logger.warning(
                        "Invalid translation for %s at index %d",
                        lang,
                        ind
                    )
                    is_valid = False
                    break
                
                ans["tgt"][lang] = trans_item[lang]
                ans["lit_tgt"][lang] = literal_item[lang]
            
            if is_valid:
                summary.append(ans)
        
        output_path = out_dir / "en2x.json"
        write_json_file(summary, str(output_path))
        logger.info("English branch completed: %d valid items -> %s", len(summary), output_path)
        
        return summary
        
    except Exception as e:
        logger.error("English branch failed: %s", str(e), exc_info=True)
        raise


def chinese_branch(out_dir: Path) -> List[dict]:
    """
    Process Chinese idioms: generate sentences and translate to multiple languages.
    
    Args:
        out_dir: Output directory path
        
    Returns:
        List of processed idiom dictionaries
    """
    logger.info("Processing Chinese idioms...")
    
    try:
        zh_data = read_json_file(INPUT_ZH_PATH)
        logger.info("Loaded %d Chinese idioms", len(zh_data))
        
        # Generate Chinese/English sentences
        logger.info("Generating Chinese sentences with idioms...")
        zh_inputs = inputs_zh_idiom_en(zh_data)
        zh_en = run_multithreading(
            lambda x: worker_schema(x, SCHEMA_ZH_EN),
            zh_inputs,
            pn=THREADS
        )
        
        # Literal translations from Chinese
        literal_map: Dict[str, List[dict]] = {}
        for lang in TARGET_LANGS:
            if lang == "Chinese":
                continue
            logger.info("Generating literal translation: Chinese -> %s", lang)
            li = inputs_literal_trans(zh_en, "Chinese", "Chinese", lang)
            literal_map[lang] = run_multithreading(worker_text, li, pn=THREADS)
        
        # General translations from English (except to English)
        trans_map: Dict[str, List[dict]] = {}
        for lang in TARGET_LANGS:
            if lang in ["Chinese", "English"]:
                continue
            logger.info("Translating English -> %s", lang)
            gi = inputs_general_trans(zh_en, "English", "English", lang)
            trans_map[lang] = run_multithreading(worker_text, gi, pn=THREADS)
        
        # Merge results
        summary: List[dict] = []
        for ind, src in enumerate(zh_data):
            item1 = zh_en[ind]
            lit_en_item = literal_map["English"][ind]
            
            if (
                not validate_item_fields(
                    item1,
                    ["id", "Chinese", "Plain_Chinese", "English"],
                    ind
                )
                or item1["id"] != src["id"]
                or lit_en_item.get("id") != src["id"]
                or not lit_en_item.get("English")
            ):
                continue
            
            ans = {
                "id": src["id"],
                "idiom": src["idiom"],
                "src": item1["Chinese"],
                "src_plain": item1["Plain_Chinese"],
                "tgt": {"English": item1["English"]},
                "lit_tgt": {"English": lit_en_item["English"]},
            }
            
            is_valid = True
            for lang in TARGET_LANGS:
                if lang in ["Chinese", "English"]:
                    continue
                
                trans_item = trans_map[lang][ind]
                lit_item = literal_map[lang][ind]
                
                if (
                    trans_item.get("id") != src["id"]
                    or lit_item.get("id") != src["id"]
                    or not trans_item.get(lang)
                    or not lit_item.get(lang)
                ):
                    logger.warning(
                        "Invalid translation for %s at index %d",
                        lang,
                        ind
                    )
                    is_valid = False
                    break
                
                ans["tgt"][lang] = trans_item[lang]
                ans["lit_tgt"][lang] = lit_item[lang]
            
            if is_valid:
                summary.append(ans)
        
        output_path = out_dir / "zh2x.json"
        write_json_file(summary, str(output_path))
        logger.info("Chinese branch completed: %d valid items -> %s", len(summary), output_path)
        
        return summary
        
    except Exception as e:
        logger.error("Chinese branch failed: %s", str(e), exc_info=True)
        raise


# =========================
# Main
# =========================
def main() -> None:
    """Main function to process English and Chinese idiom branches."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", OUTPUT_DIR)
        
        english_branch(OUTPUT_DIR)
        chinese_branch(OUTPUT_DIR)
        
        logger.info("All processing completed successfully")
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()