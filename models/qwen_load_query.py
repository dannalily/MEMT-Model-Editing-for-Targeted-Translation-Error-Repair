import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoModelForCausalLM, AutoTokenizer

class qwenM:
    def __init__(self, model):

        self.model = AutoModelForCausalLM.from_pretrained(model,torch_dtype="auto",device_map="auto")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def query(self, system_prompt, user_prompt, max_new_tokens = 200, num_beams=5, do_sample=False, temperature=None,top_p=None,top_k=None):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        inp = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        encoded_inp = self.tokenizer([inp], return_tensors="pt").to(self.model.device)
   
        generated_ids = self.model.generate(
            **encoded_inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(encoded_inp.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


# model = qwenM('Qwen/Qwen2.5-72B-Instruct')
# system_prompt='''You are tasked with generating structured outputs for Chinese idioms. For each given idiom and its meaning in Chinese and English, follow these steps:
# 1. Chinese Sentence with Idiom: 
#       - Create a extreme short sentence (≤ 15 words) in Chinese that uses the idiom naturally and correctly.  
#     - The sentence must be clear, self-contained, and unambiguous. 
#     - The sentence must be concise and **must not add extra context or redundant explanation beyond the idiom’s meaning**.  
# 2. Plain Chinese Sentence
#     - Based on the provided Chinese meaning of the idiom, write a clear, simple, idiom-free parallel Chinese sentence.
#     - This sentence should not use any idiom and must directly express the same meaning.
# 3. Plain English Translation
#      - Based on the parallel Chinese sentence and the given English meaning, provide a clear, simple English sentence.
#      - The English must reflect the idiomatic meaning, without using any idiom.

# Output Format:
# - Return a valid JSON object with the following keys only:
# {
#   "Chinese": "<chinese sentence with idiom>",
#   "Plain_Chinese": "Plain Chinese Sentence>",
#   "English": "<plain English translation>"
# }'''

# user_prompt = '''Chinese Idiom: 一板三眼
# Chinese Meaning: 形容人严肃认真，要求严厉
# English Meaning: meticulous and strict in one's work or behavior'''
# print(model.query(system_prompt,user_prompt, max_new_tokens = 1024, num_beams=1, do_sample=True, temperature=0.7,top_p=0.8,top_k=20))
