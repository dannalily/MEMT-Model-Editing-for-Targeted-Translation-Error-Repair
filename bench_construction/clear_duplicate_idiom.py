"""
Module for deduplicating idioms in IdiomKB datasets.

This script removes duplicate idioms from English and Chinese idiom datasets
based on normalized idiom text.
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Make parent importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    read_json_file,
    write_json_file,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
INPUT_EN_PATH = "IdiomKB/en_idiom_meaning.json"
OUTPUT_EN_PATH = "IdiomKB/en_idiom_meaning_dedupe.json"
INPUT_ZH_PATH = "IdiomKB/zh_idiom_meaning.json"
OUTPUT_ZH_PATH = "IdiomKB/zh_idiom_meaning_dedupe.json"


def normalize_idiom(idiom_text: str) -> str:
    """
    Normalize idiom text for comparison.
    
    Args:
        idiom_text: Raw idiom text to normalize
        
    Returns:
        Normalized idiom text (lowercase, no underscores, single spaces)
    """
    return " ".join(
        idiom_text.lower().strip().replace("_", " ").split()
    )


def dedupe_idiom(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate idioms from dataset based on normalized idiom text.
    
    Args:
        data: List of idiom dictionaries containing 'id' and 'idiom' keys
        
    Returns:
        Deduplicated list of idiom dictionaries
    """
    idiom_seen = set()
    new_data = []
    
    for item in data:
        try:
            normalized_idiom = normalize_idiom(item['idiom'])
            
            if normalized_idiom in idiom_seen:
                continue
            
            new_data.append(item)
            idiom_seen.add(normalized_idiom)
            
        except KeyError as e:
            logger.error("Missing required key '%s' in item: %s", e, item)
            raise
        except Exception as e:
            logger.error("Error processing item: %s", item, exc_info=True)
            raise
    
    return new_data


def process_idiom_file(
    input_path: str,
    output_path: str,
    language: str
) -> None:
    """
    Process idiom file by reading, deduplicating, and writing results.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        language: Language identifier for logging (e.g., 'English', 'Chinese')
    """
    try:
        data = read_json_file(input_path)
        logger.info("%s - Before dedupe: %d", language, len(data))
        
        deduplicated_data = dedupe_idiom(data)
        logger.info("%s - After dedupe: %d", language, len(deduplicated_data))
        
        write_json_file(deduplicated_data, output_path)
        
    except Exception as e:
        logger.error("Error processing %s idiom file: %s", language, str(e), exc_info=True)
        raise


def main() -> None:
    """Main function to deduplicate English and Chinese idiom datasets."""
    try:
        process_idiom_file(INPUT_EN_PATH, OUTPUT_EN_PATH, "English")
        process_idiom_file(INPUT_ZH_PATH, OUTPUT_ZH_PATH, "Chinese")
        logger.info("Deduplication completed successfully")
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()