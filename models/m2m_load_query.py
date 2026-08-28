import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class m2mM:
    def __init__(self, model):

        self.model = M2M100ForConditionalGeneration.from_pretrained(model,torch_dtype="auto",device_map="cuda")
        self.model.eval()
        self.tokenizer = M2M100Tokenizer.from_pretrained(model)

    def input_organize(self, src_lang, inp):
        self.tokenizer.src_lang = src_lang
        encoded_inp = self.tokenizer(inp, return_tensors="pt")
        return encoded_inp

    def query(self, src_lang, tar_lang, inp, num_beams=5, max_new_tokens=200, do_sample=False):
        self.tokenizer.src_lang = src_lang
        encoded_inp = self.tokenizer(inp, return_tensors="pt").to("cuda")
        generated_tokens = self.model.generate(**encoded_inp, forced_bos_token_id=self.tokenizer.get_lang_id(tar_lang), num_beams = num_beams, max_new_tokens = max_new_tokens,do_sample=do_sample)
        response = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return response
