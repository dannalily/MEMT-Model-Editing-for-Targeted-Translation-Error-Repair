import torch
import numpy as np
from tqdm import tqdm
options = ["A", "B", "C", "D"]
sys_prompt = '''You are a helpful assistant that answers multiple-choice questions. For each question, you must output ONLY the single correct option letter (A, B, C, or D). Do not output explanations, punctuation, or any other text.
'''

def format_example(question, choices, tokenizer,model_name):
    assert model_name in ['Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']

    prompt = question
    for j in range(len(choices)):
        prompt += "\n{}. {}".format(options[j], choices[j])
    prompt += "\nAnswer:"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

@torch.no_grad()
def eval_mmlu(model, tokenizer, eval_datas, model_name):
    questions = [item['question'] for item in eval_datas]
    choices = [item['choices'] for item in eval_datas]
    answers = [item['answer'] for item in eval_datas]

    cors = []
    preds = []
    for i in range(len(questions)): #batch_size
        prompt = format_example(questions[i], choices[i],tokenizer,model_name)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        logits = model(
            input_ids=input_ids
        ).logits[0,-1] #bs:seq:vocab_size

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        cor = 1 if np.argmax(probs) == answers[i] else 0
        cors.append(cor)
        preds.append(pred)
    return cors, preds


if __name__ == "__main__":
    from utils import read_json_file, write_json_file
    from transformers import AutoTokenizer, AutoModelForCausalLM

    data =read_json_file('MEMT/locality_MMMLU_shuffle1000.json')
    LANGS = ['Chinese','English','French','German','Arabic','Japanese']
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    model_name_ans = "Qwen2-5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda:0')
    tok = AutoTokenizer.from_pretrained(model_name)
    pred_dict = {}
    eval_dict = {}
    for lang in LANGS:
        cors, preds = eval_mmlu(model, tok, data[lang], model_name)
        pred_dict[lang] = preds
        eval_dict[lang] = cors

    write_json_file(pred_dict, f'MEMT/locality_MMMLU_pred/{model_name_ans}.json')
    write_json_file(eval_dict, f'MEMT/locality_MMMLU_eval/{model_name_ans}.json')



