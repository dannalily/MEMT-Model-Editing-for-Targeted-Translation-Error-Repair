"""Translation generation script for ParaIdiomSent benchmark."""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Configuration
CONFIG = {
    'random_seed': 42,
    'output_base_dir': 'bench_construction/MEMT/ParaIdiomSent_trans'
}


def preparse_gpus():
    """Pre-parse GPU arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default="0")
    args, _ = parser.parse_known_args()
    return args.gpus


# Set CUDA before importing GPU libraries
os.environ["CUDA_VISIBLE_DEVICES"] = preparse_gpus()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import read_json_file, write_json_file, LANGS, LANGS_TO_BCP47
from prompt import PROMPT_DICT


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_translator(translator_name: str, translator_id: str):
    """Load translator model."""
    if translator_name == "nllb":
        from models.nllb_load_query import nllbM as TranslatorClass
    elif translator_name == "m2m":
        from models.m2m_load_query import m2mM as TranslatorClass
    elif translator_name == "qwen-instruct":
        from models.qwen_load_query import qwenM as TranslatorClass
    else:
        raise ValueError(f"Unsupported translator: {translator_name}")
    
    return TranslatorClass(translator_id)


def generate_translation(source_lang: str, target_lang: str, source_text: str, 
                        translator, translator_name: str) -> str:
    """Generate translation for source text."""
    if translator_name == 'qwen-instruct':
        task = 'GENERAL_TRANS'
        sys_prompt = PROMPT_DICT[task]['sys'].format(
            SLang=LANGS[source_lang], TLang=LANGS[target_lang]
        )
        user_prompt = PROMPT_DICT[task]['user'].format(
            SLang=LANGS[source_lang], Source=source_text, TLang=LANGS[target_lang]
        )
        return translator.query(sys_prompt, user_prompt)
        
    elif translator_name == 'nllb':
        return translator.query(
            LANGS_TO_BCP47[source_lang], 
            LANGS_TO_BCP47[target_lang], 
            source_text
        )
        
    elif translator_name == 'm2m':
        return translator.query(source_lang, target_lang, source_text)
    
    else:
        raise ValueError(f"Unsupported translator: {translator_name}")


def main():
    """Main execution."""
    set_seed(CONFIG['random_seed'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--translator_id", type=str, required=True)
    parser.add_argument("--translator_name", type=str, required=True)
    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument("--target_lang", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    
    # Setup output path
    model_name = args.translator_id.split('/')[-1].replace('.', '-').replace("_", "-")
    output_dir = Path(CONFIG['output_base_dir']) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.source_lang}_{args.target_lang}.json"
    
    print(f"{model_name}_{args.source_lang}_{args.target_lang}")
    
    # Load translator and data
    translator = load_translator(args.translator_name, args.translator_id)
    data = read_json_file(args.data_file)
    
    # Generate translations
    results = []
    for item in tqdm(data):
        translation = generate_translation(
            args.source_lang, 
            args.target_lang, 
            item['src'], 
            translator, 
            args.translator_name
        )
        results.append(translation)
    
    write_json_file(results, str(output_path))


if __name__ == "__main__":
    main()