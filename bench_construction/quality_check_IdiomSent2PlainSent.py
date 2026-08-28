"""
Module for paraphrasing idiom sentences and evaluating with BLEU scores.

This script uses Claude batch inference to paraphrase idiomatic sentences
by replacing idioms with plain expressions, then evaluates the results
against reference plain sentences using BLEU scores.
"""

import logging
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import load_csv_file, read_jsonl_batch_inference
from models.claudeBatch_load_query import ClaudeBatchM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
MODEL_NAME = "us.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_TOKENS = 10000
TEMPERATURE = 0.7

INPUT_CSV_PATH = "bench_construction/data_cleaned.csv"
S3_BUCKET = 'dann02'
LOCAL_INPUT_FILE = 'bench_construction/intermediate_out/batch_cache/pie_paraphrase.jsonl'
LOCAL_OUTPUT_FILE = 'bench_construction/intermediate_out/batch_cache/pie_paraphrase_out.jsonl'

SYSTEM_PROMPT = '''You are tasked with generating structured outputs for English idioms.
For each given idiom, its meaning in English, and a sentence containing the idiom:
    - Rewrite the idiom sentence by replacing the idiom with a plain, direct expression of its meaning.
    - Make only minimal changes to the sentence structure. Keep wording and grammar as close to the idiom sentence as possible.
    - Do not introduce new context or elaboration.

Output Format:
Return a valid JSON object with the following keys only:
{
  "Plain_English": "<Plain English sentence>"
}'''

USER_PROMPT_TEMPLATE = '''English Idiom: {Idiom}
English Meaning: {Meaning}
Idiomatic Sentence: {Sent}'''

SCHEMA = {
    "type": "object",
    "properties": {
        "Plain_English": {"type": "string"},
    },
    "required": ["Plain_English"],
    "additionalProperties": False,
}


def prepare_prompts(data: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Prepare system and user prompts from input data.
    
    Args:
        data: List of idiom dictionaries
        
    Returns:
        Tuple of (system_prompts, user_prompts)
    """
    system_prompts = [SYSTEM_PROMPT] * len(data)
    user_prompts = [
        USER_PROMPT_TEMPLATE.format(
            Idiom=item['Idiom'],
            Meaning=item['Sense'],
            Sent=item['Idiomatic_Sent']
        )
        for item in data
    ]
    return system_prompts, user_prompts


def run_batch_paraphrase(
    system_prompts: List[str],
    user_prompts: List[str]
) -> List[str]:
    """
    Run batch inference to paraphrase idiom sentences.
    
    Args:
        system_prompts: List of system prompts
        user_prompts: List of user prompts
        
    Returns:
        List of paraphrased sentences
    """
    logger.info("Preparing batch inference for %d items", len(user_prompts))
    
    model = ClaudeBatchM(MODEL_NAME)
    
    model.prepare_input_config(
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        local_input_file=LOCAL_INPUT_FILE,
        s3_bucket=S3_BUCKET,
        s3_key='other/input/pie_paraphrase.jsonl',
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        json_schema=SCHEMA
    )
    
    logger.info("Submitting batch job...")
    job_arn = model.submit_and_wait(
        job_name='pieParaphrase',
        s3_bucket=S3_BUCKET,
        s3_input_dir='other/input',
        s3_input_file='pie_paraphrase.jsonl',
        s3_output_dir='other/output',
        local_output_file=LOCAL_OUTPUT_FILE
    )
    
    logger.info("Batch job completed: %s", job_arn)
    
    # Extract paraphrases
    results = read_jsonl_batch_inference(LOCAL_OUTPUT_FILE, structured_out=True)
    paraphrases = [item['pred']['Plain_English'] for item in results]
    
    logger.info("Extracted %d paraphrases", len(paraphrases))
    return paraphrases


def calculate_bleu_scores(
    references: List[str],
    hypotheses: List[str]
) -> float:
    """
    Calculate BLEU scores between reference and hypothesis sentences.
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
        
    Returns:
        Average BLEU score
    """
    smooth_fn = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(score)
    
    avg_score = sum(bleu_scores) / len(bleu_scores)
    logger.info("Average BLEU score: %.4f", avg_score)
    
    return avg_score


def main() -> None:
    """Main function to run paraphrasing and evaluation."""
    try:
        logger.info("Starting idiom sentence paraphrasing evaluation")
        
        # Load data
        logger.info("Loading data from: %s", INPUT_CSV_PATH)
        data = load_csv_file(INPUT_CSV_PATH)
        logger.info("Loaded %d items", len(data))
        
        # Extract sources
        idiomatic_sentences = [item['Idiomatic_Sent'] for item in data]
        literal_sentences = [item['Literal_Sent'] for item in data]
        
        # Prepare prompts
        system_prompts, user_prompts = prepare_prompts(data)
        
        # Run batch paraphrase
        paraphrases = run_batch_paraphrase(system_prompts, user_prompts)
        
        # Calculate BLEU scores
        logger.info("Calculating BLEU scores...")
        avg_bleu = calculate_bleu_scores(literal_sentences, paraphrases)
        
        print(f"\nBLEU Score (Claude paraphrase vs PIE plain): {avg_bleu:.4f}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()