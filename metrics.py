import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import sys
sys.path.append("src/metricx")
import io, contextlib, logging
import torch
import transformers
from tqdm import tqdm
from models.claudeBatch_load_query import ClaudeBatchM
from models.claude_load_query import ClaudeM
from prompt import PROMPT_DICT
from utils import read_jsonl_batch_inference

def _silence_lightning_rank_zero():
    """Silence all Lightning rank-zero logging output for cleaner console."""
    try:
        # lightning 2.x
        import lightning.pytorch.utilities.rank_zero as rz2
        def _noop(*a, **k): pass
        rz2.rank_zero_info = _noop
        rz2.rank_zero_debug = _noop
        rz2.rank_zero_warn = _noop
        rz2.rank_zero_print = _noop
    except Exception:
        pass
    try:
        # pytorch_lightning 1.x
        import pytorch_lightning.utilities.rank_zero as rz1
        def _noop(*a, **k): pass
        rz1.rank_zero_info = _noop
        rz1.rank_zero_debug = _noop
        rz1.rank_zero_warn = _noop
        # Pretend we are not rank0 to avoid printing
        rz1.rank_zero_only.rank = 1
    except Exception:
        pass

@contextlib.contextmanager
def _quiet_logs():
    """Temporarily suppress all stdout/stderr and logging output."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    prev_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield
    finally:
        logging.disable(prev_disable)

class metricx23Scorer:
    """
    Scorer for google/metricx-23-xl-v2p0 model.
    Note: Lower score is better, typical range is [0, 25].
    """
    def __init__(self, model_name_or_path='google/metricx-23-xl-v2p0', tokenizer_name_or_path='google/mt5-xl', max_input_length=1024,is_qe=False,verbose=True):
        from metricx23 import models  
        ## note： metricx lower is better , range [0,25]

        self.models = models
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = self.models.MT5ForRegression.from_pretrained(model_name_or_path,device_map="cuda")
        self.model.eval()
        self.max_input_length = max_input_length
        self.is_qe = is_qe
        self.verbose =verbose

    @torch.no_grad()
    def batch_score(self, translations, referencesORsources, batch_size=8):
        """Compute MetricX scores for multiple translation-reference pairs."""
        assert len(translations) == len(referencesORsources)
        if self.is_qe:
            data = [f"candidate: {t} source: {r}" for t, r in zip(translations, referencesORsources)]
        else:
            data = [f"candidate: {t} reference: {r}" for t, r in zip(translations, referencesORsources)]
        scores = []
        if self.verbose:
            for i in tqdm(range(0, len(data), batch_size)):
                chunk = data[i:i+batch_size]
                enc = self.tokenizer(
                    chunk,
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )

                input_ids = enc["input_ids"][:, :-1].to("cuda")
                attention_mask = enc["attention_mask"][:, :-1].to("cuda")
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out.predictions.squeeze(-1).detach().cpu().tolist()
                if isinstance(preds, float):
                    preds = [preds]
                scores.extend([float(x) for x in preds])
        else:
            for i in range(0, len(data), batch_size):
                chunk = data[i:i+batch_size]
                enc = self.tokenizer(
                    chunk,
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )

                input_ids = enc["input_ids"][:, :-1].to("cuda")
                attention_mask = enc["attention_mask"][:, :-1].to("cuda")
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out.predictions.squeeze(-1).detach().cpu().tolist()
                if isinstance(preds, float):
                    preds = [preds]
                scores.extend([float(x) for x in preds])

        return scores

    def score(self, translation, reference):
        """Single-pair scoring."""
        return self.batch_score([translation], [reference], batch_size=1)[0]


class cometScorer:
    """
    Scorer for Unbabel/wmt22-comet-da.
    Note: Higher score is better, typical range is [0, 1].
    """
    def __init__(self, model_name_or_path='Unbabel/wmt22-comet-da'):
        from comet import download_model, load_from_checkpoint
        ## note： higher is better , range [0,1]

        model_path = download_model(model_name_or_path)
        self.model = load_from_checkpoint(model_path)
        _silence_lightning_rank_zero() 

    def batch_score(self, translations, references, sources, batch_size=8):
        """Compute COMET scores for multiple triplets (src, mt, ref)."""
        assert len(translations) == len(references) == len(sources), "Mismatch in list lengths"
        data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(sources, translations, references)]
        scores = []
        for i in tqdm(range(0, len(data), batch_size)):
            chunk = data[i:i+batch_size]
            with _quiet_logs():
                out = self.model.predict(
                    chunk,
                    batch_size=len(chunk),
                    gpus=1,
                    progress_bar=False,
                    num_workers=0,
                    accelerator="cuda"
                )
            scores.extend([float(x) for x in out.scores])
        return scores


    def score(self, translation, reference, source):
        """Single-triplet scoring."""
        return self.batch_score([translation], [reference], [source], batch_size=1)[0]
    
class bleurtScorer:
    """
    Scorer for lucadiliello/BLEURT-20.
    Note: Higher score is better, typical range ~[-1, 1+] but can exceed.
    """
    def __init__(self, model_name_or_path='lucadiliello/BLEURT-20'):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        self.model = BleurtForSequenceClassification.from_pretrained(model_name_or_path,device_map="cuda")
        self.model.eval()
        self.tokenizer = BleurtTokenizer.from_pretrained(model_name_or_path)

    @torch.no_grad()
    def batch_score(self, translations, references, batch_size=8):
        """Compute BLEURT scores for multiple translation-reference pairs."""
        assert len(translations) == len(references), "Length mismatch between translations and references"
        scores = []
        for i in tqdm(range(0, len(translations), batch_size)):
            refs = references[i:i+batch_size]
            mts = translations[i:i+batch_size]
            inputs = self.tokenizer(refs, mts, max_length=512,truncation=True, padding=True,  return_tensors='pt')
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            logits = self.model(**inputs).logits.flatten().detach().cpu().tolist()
            scores.extend([float(x) for x in logits])
        return scores

    def score(self, translation, reference):
        return self.batch_score([translation], [reference], batch_size=1)[0]


class claudeScorer():
    def __init__(self, mode, model_name='us.anthropic.claude-sonnet-4-20250514-v1:0'):
        SCORE_SCHEMA = {
            'SCALE':{
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "A score between 0 and 100, can be a decimal."
                    }
                },
                "required": ["score"],
                "additionalProperties": False
            },
            'BINARY':{
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "0 = unacceptable, 1 = acceptable"
                    }
                },
                "required": ["score"],
                "additionalProperties": False
            }
        }
        self.sys_prompt_temp = PROMPT_DICT[f'TRANS_{mode}_EVAL']['sys']
        self.user_prompt_temp =  PROMPT_DICT[f'TRANS_{mode}_EVAL']['user']
        self.out_schema = SCORE_SCHEMA[mode]
        self.model_name = model_name

    def input_process(self, translations, references, slang, tlang):
        assert len(translations) == len(references)
        sys_prompts=[]
        user_prompts=[]
        for trans, ref in list(zip(translations, references)):
            sys_prompts.append(self.sys_prompt_temp.format(SLang = slang, TLang = tlang))
            user_prompts.append(self.user_prompt_temp.format(TLang = tlang, Reference = ref, Target=trans))

        return  sys_prompts,user_prompts
    
    def score_one_by_one(self, translations, references, slang, tlang):
        results=[]
        sys_prompts, user_prompts = self.input_process(translations, references, slang, tlang)
        runner = ClaudeM(self.model_name)
        for system_prompt, user_prompt in tqdm(list(zip(sys_prompts, user_prompts))):
            _, body = runner.query(system_prompt, user_prompt, json_schema = self.out_schema)
            # print(body["structured_output"]['score'])
            results.append(body["structured_output"]['score'])
        return results

    def batch_score(self, translations, references, slang, tlang, local_input_file,local_output_file, s3_bucket, s3_input_dir, s3_input_file, s3_output_dir, job_name):
        sys_prompts, user_prompts = self.input_process(translations, references, slang, tlang)
        runner = ClaudeBatchM(self.model_name)
        runner.prepare_input_config(
            system_prompts=sys_prompts,
            user_prompts=user_prompts,
            local_input_file=local_input_file,
            s3_bucket=s3_bucket,
            s3_key=f"{s3_input_dir}/{s3_input_file}",
            temperature = 0,
            json_schema = self.out_schema
            )
        
        runner.submit_and_wait(
            job_name = job_name,
            s3_bucket = s3_bucket,
            s3_input_dir = s3_input_dir,
            s3_input_file = s3_input_file,
            s3_output_dir = s3_output_dir,
            local_output_file = local_output_file
        )

        results = read_jsonl_batch_inference(local_output_file, structured_out=True)

        return [float(item['pred']['score']) for item in results]
