from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from editors.GRACE.GRACE import GRACE
from grace_hparams import GraceHyperParams
from editors.GRACE.utils import tokenize

def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        system_prompt: str,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)
    device = torch.device(f'cuda:{hparams.device}')
    editor = GRACE(model=model, config=hparams, device=device)
    tokens = tokenize(request, tokenizer=tok, device=device,system_prompt=system_prompt,hparams=hparams)
    editor.edit(config=hparams, tokens=tokens,edit_id=request['target_new'])
            
    weights_copy = editor.reset_layer


    return editor, weights_copy