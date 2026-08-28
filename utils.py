import json 
from typing import Callable, Union, Optional
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm
import csv
import time
import random
import boto3

LANGS = {
    "en":"English",
    "zh":"Chinese",
    "ja":"Japanese",
    "fr":"French",
    "de":"German",
    "ar":"Arabic"
}
LANGS_TO_BCP47 = {
    "en":"eng_Latn",
    "zh":"zho_Hans",
    "ja":"jpn_Jpan",
    "fr":"fra_Latn",
    "de":"deu_Latn",
    "ar":"arb_Arab"
}


# =========================
# File helpers
# =========================
def write_json_file(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def read_jsonl_batch_inference(filepath,structured_out=True):
    preds=[]
    with open(filepath, 'r') as f:
        for line in f:
            response = json.loads(line)
            if structured_out:
                preds.append({'recordId':int(response['recordId']),
                            'pred': response['modelOutput']['content'][0]['input'] })
            else:
                preds.append({'recordId':int(response['recordId']),
                            'pred': response['modelOutput']['content'][0]['text'] })

    preds.sort(key=lambda r: r['recordId'])
    return preds
    
def load_csv_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  
        return list(reader)

def write_jsonl(path, records, append=False):
    """Write iterable of dict-like records to a JSONL file."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")  # one object per line

def read_jsonl(path):
    """Stream records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_file2s3(localFile,bucket,remotePath):
    s3 = boto3.client("s3")
    s3.upload_file(localFile,bucket,remotePath)

def download_fileFs3(bucket, remoteFile, localFile):
    s3 = boto3.client("s3")
    s3.download_fileobj(bucket, remoteFile, localFile)

# =========================
# Concurrency
# =========================

def run_multithreading(process_func: Callable, inp_list: Union[list, range], pn: Optional[int] = None) -> list:
    """
    Run the process_func in multithreading mode over all inputs.
    @param process_func: a function that returns (index, result)
    @param inp_list: list of inputs (should be indexable)
    @param pn: number of threads
    @return: list of results, ordered by original input order
    """
    def get_cpu_counts() -> int:
        return max(mp.cpu_count(), 4)

    if pn is None:
        pn = get_cpu_counts() * 5
    res = [None] * len(inp_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=pn) as executor:
        for idx, feat in tqdm(executor.map(process_func, inp_list), total=len(inp_list), desc="Processing"):
            res[idx] = feat
    return res

# =========================
# Retry helpers
# =========================

def sleep_with_backoff(try_num: int = 10, base_seconds: float = 2.0) -> None:
    delay = base_seconds * (2 ** try_num)
    time.sleep(delay * (0.5 + random.random()))


def query_with_retries(generator, sys_prompt: str, user_prompt: str, *, schema: dict | None = None,
                       max_retries: int = 8, temperature: float = 0.7, max_tokens: int = 10000):
    """
    Query ClaudeM with retries and exponential backoff.
    Returns (text, response) or (None, None).
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            if schema:
                text, resp = generator.query(
                    sys_prompt, user_prompt,
                    json_schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                text, resp = generator.query(
                    sys_prompt, user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return text.strip(), resp
        except Exception as e:
            last_err = e
            sleep_with_backoff(attempt)
    print(f"[warn] query failed: {last_err}")
    return None, None



