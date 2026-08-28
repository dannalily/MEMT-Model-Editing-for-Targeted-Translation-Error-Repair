"""Identify idiom spans in sentences using Claude."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import read_json_file, write_json_file
from models.claude_load_query import ClaudeM
from prompt import PROMPT_DICT


# Configuration
CONFIG = {
    'source_langs': ['en', 'zh'],
    'claude_model': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
    'max_retries': 10,
    'base_paths': {
        'para_data': 'bench_construction/MEMT/ParaIdiomSent',
        'output': 'bench_construction'
    }
}


def extract_idiom_span_with_claude(idiom: str, sentence: str, source_lang: str, 
                                   model: ClaudeM, max_retries: int = 10) -> str:
    """
    Extract idiom span from sentence using Claude.
    
    Args:
        idiom: The idiom to find
        sentence: The sentence containing the idiom
        source_lang: Source language code
        model: Claude model instance
        max_retries: Maximum number of retry attempts
    
    Returns:
        Extracted idiom span
        
    Raises:
        ValueError: If extraction fails after max_retries
    """
    sys_prompt = PROMPT_DICT[f'{source_lang.upper()}_IDIOM_SPAN_IDENTIFY']['sys']
    user_prompt = PROMPT_DICT[f'{source_lang.upper()}_IDIOM_SPAN_IDENTIFY']['user'].format(
        Idiom=idiom, 
        SENTENCE=sentence
    )
    
    for attempt in range(max_retries):
        try:
            response_text, _ = model.query(sys_prompt, user_prompt)
            extracted_span = response_text.strip()
            
            if extracted_span and extracted_span in sentence:
                return extracted_span
            else:
                raise ValueError(f"Extracted span '{extracted_span}' not found in sentence")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed after {max_retries} attempts: {str(e)}")
            continue
    
    raise ValueError(f"Failed to extract idiom span after {max_retries} attempts")


def identify_idiom_span(item: Dict, source_lang: str, model: ClaudeM) -> str:
    """
    Identify idiom span in a single item.
    
    Args:
        item: Data item containing idiom and sentence
        source_lang: Source language code
        model: Claude model instance
    
    Returns:
        Identified idiom span
    """
    idiom = item["idiom"]
    sentence = item["src"]
    
    # Direct match - idiom appears exactly in sentence
    if idiom in sentence:
        return idiom
    
    # Use Claude to extract the span
    return extract_idiom_span_with_claude(idiom, sentence, source_lang, model, CONFIG['max_retries'])


def process_language(source_lang: str) -> Tuple[Dict[str, str], List[int]]:
    """
    Process all items for a source language.
    
    Args:
        source_lang: Source language code
    
    Returns:
        Tuple of (idiom_span_dict, failed_indices)
    """
    # Load data
    para_data_file = f"{CONFIG['base_paths']['para_data']}/{source_lang}2x.json"
    para_data = read_json_file(para_data_file)
    
    # Initialize Claude model once
    model = ClaudeM(CONFIG['claude_model'])
    
    idiom_spans = {}
    failed_indices = []
    
    for idx, item in enumerate(tqdm(para_data, desc=f"Processing {source_lang}")):
        try:
            span = identify_idiom_span(item, source_lang, model)
            idiom_spans[item['id']] = span
        except ValueError as e:
            print(f"\nFailed on item {idx}: {item}")
            print(f"Error: {e}")
            failed_indices.append(idx)
    
    return idiom_spans, failed_indices


def main():
    """Main execution."""
    for source_lang in CONFIG['source_langs']:
        print(f"\n{'='*50}")
        print(f"Processing language: {source_lang}")
        print(f"{'='*50}")
        
        idiom_spans, failed_indices = process_language(source_lang)
        
        # Save results
        output_file = f"{CONFIG['base_paths']['output']}/idiom_span_{source_lang}2x.json"
        write_json_file(idiom_spans, output_file)
        
        print(f"\n{source_lang} - Failed indices: {failed_indices}")
        print(f"{source_lang} - Successfully processed: {len(idiom_spans)}/{len(idiom_spans) + len(failed_indices)}")
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()