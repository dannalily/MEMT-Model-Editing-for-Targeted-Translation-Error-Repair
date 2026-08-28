SENT_IDIOM_ZH_EN_SYS = '''You are tasked with generating structured outputs for Chinese idioms. For each given idiom and its meaning in Chinese and English, follow these steps:
1. Chinese Sentence with Idiom: 
    - Create a extreme short sentence (≤ 15 words) in Chinese that uses the idiom naturally and correctly.  
    - The sentence must be clear, self-contained, and unambiguous. 
    - The sentence must be concise and **must not add extra context or redundant explanation beyond the idiom’s meaning**.  
2. Plain Chinese Sentence
    - Rewrite the idiom sentence by replacing the idiom with a plain, direct expression of its meaning.
    - Make only minimal changes to the sentence structure. Keep wording and grammar as close to the idiom sentence as possible.
    - Do not introduce new context or elaboration.
3. Plain English Translation
     - Based on the parallel Chinese sentence and the given English meaning, provide a clear, simple English sentence.
     - The English must reflect the idiomatic meaning, without using any idiom.

Output Format:
- Return a valid JSON object with the following keys only:
{
  "Chinese": "<chinese sentence with idiom>",
  "Plain_Chinese": "Plain Chinese Sentence>",
  "English": "<plain English translation>"
}
'''

SENT_IDIOM_ZH_EN_USER = '''Chinese Idiom: {Idiom}
Chinese Meaning: {ZH_Meaning}
English Meaning: {EN_Meaning}'''

############
SENT_IDIOM_EN_EN_SYS = '''You are tasked with generating structured outputs for English idioms.
For each given idiom and its meaning in English, follow these steps:
1. English Sentence with Idiom
    - Create an extremely short sentence (≤ 15 words) in English that uses the idiom naturally and correctly.
    - The sentence must be clear, self-contained, and unambiguous.
    - The sentence must be concise and must not add extra context or redundant explanation beyond the idiom’s meaning.
2. Plain English Sentence
    - Rewrite the idiom sentence by replacing the idiom with a plain, direct expression of its meaning.
    - Make only minimal changes to the sentence structure. Keep wording and grammar as close to the idiom sentence as possible.
    - Do not introduce new context or elaboration.

Output Format:
Return a valid JSON object with the following keys only:
{
  "English": "<English sentence with idiom>",
  "Plain_English": "<Plain English sentence>"
}'''

SENT_IDIOM_EN_EN_USER = '''English Idiom: {Idiom}
English Meaning: {Meaning}'''

############
GENERAL_TRANS_SYS = '''Translate text from {SLang} to {TLang}.
    - The translation must be accurate, fluent, and preserve the original meaning.
    - Do not add explanations, notes, or extra content.
    - Output only the translated text.'''

GENERAL_TRANS_USER = '''{SLang}: {Source}
{TLang}: '''

############
LITERAL_SYS_PROMPT = '''Produce a bad example by generating an unnatural, strictly literal word-for-word translation from {SLang} to {TLang}.
    - The translation must deliberately reflect a word-for-word mapping, ignoring idioms and natural fluency.
    - Do not add explanations, notes, or extra content.
    - Output only the literal translation text.'''

LITERAL_USER_PROMPT ='''{SLang}: {Source}
{TLang} Literal Translation:'''

############
SCALE_TRANS_EVAL_SYS_PROMPT = '''Score the following {SLang} to {TLang} translation with respect to the human reference on a continuous scale from 0 to 100, where score of 0 means "no meaning preserved" and score of 100 means "perfect meaning and grammar".'''

SCALE_TRANS_EVAL_USER_PROMPT = '''{TLang} human reference: {Reference}
{TLang} translation: "{Target}"
Score (0-100):'''

############
BINARY_TRANS_EVAL_SYS_PROMPT = '''Score the following {SLang} to {TLang} translation with respect to the human reference on a binary scale, where a score of 0 means "unacceptable" and a score of 1 means "acceptable".'''

BINARY_TRANS_EVAL_USER_PROMPT = '''{TLang} human reference: {Reference}
{TLang} translation: "{Target}"
Score (0 or 1):'''

############
GENERALITY_SAMPLE_ZH_EN_SYS_PROMPT = '''You are tasked with generating structured outputs for Chinese idioms. For each given idiom and its meaning in Chinese and English, follow these steps:
1. Chinese Sentence with Idiom: 
    - Create 10 distinct short sentences in Chinese that uses the idiom naturally and correctly.  
    - Each sentence must be clear, self-contained, and unambiguous. 
    - Each sentence must be concise and **must not add extra context or redundant explanation beyond the idiom’s meaning**.  
    - Sentences must vary in subject, tone, and context to avoid repetition.
2. Plain Chinese Sentence
    - Rewrite each idiom sentence by replacing the idiom with a plain, direct expression of its meaning.
    - Make only minimal changes to the sentence structure. Keep wording and grammar as close to the idiom sentence as possible.
    - Do not introduce new context or elaboration.
3. Plain English Translation
     - Based on the parallel Chinese sentence and the given English meaning, provide a clear, simple English sentence.
     - The English must reflect the idiomatic meaning, without using any idiom.

Output Format:
- Return only a valid JSON array of 10 objects, each with the following keys only:
{
  "Chinese": "<chinese sentence with idiom>",
  "Plain_Chinese": "Plain Chinese Sentence>",
  "English": "<plain English translation>"
}
Note: The user may provide one example object. You must generate 10 new objects that strictly do not duplicate the user’s provided example.'''

GENERALITY_SAMPLE_ZH_EN_USER_PROMPT = '''Chinese Idiom: {Idiom}
Chinese Meaning: {ZH_Meaning}
English Meaning: {EN_Meaning}

Example: 
{{
    "Chinese": {Idiom_sent},
    "Plain_Chinese": {Plain_Idiom_sent},
    "English": {Translation}
}}'''

############
GENERALITY_SAMPLE_EN_EN_SYS_PROMPT = '''You are tasked with generating structured outputs for English idioms.
For each given idiom and its meaning in English, follow these steps:
1. English Sentence with Idiom
    - Create 10 distinct short sentence in English that uses the idiom naturally and correctly.
    - Each sentence must be clear, self-contained, and unambiguous.
    - Each sentence must be concise and must not add extra context or redundant explanation beyond the idiom’s meaning.
    - Sentences must vary in subject, tone, and context to avoid repetition.
2. Plain English Sentence
    - Rewrite each idiom sentence by replacing the idiom with a plain, direct expression of its meaning.
    - Make only minimal changes to the sentence structure. Keep wording and grammar as close to the idiom sentence as possible.
    - Do not introduce new context or elaboration.

Output Format:
- Return only a valid JSON array of 10 objects, each with the following keys only:
{
  "English": "<English sentence with idiom>",
  "Plain_English": "<Plain English sentence>"
}
Note: The user may provide one example object. You must generate 10 new objects that strictly do not duplicate the user’s provided example.'''

GENERALITY_SAMPLE_EN_EN_USER_PROMPT = '''English Idiom: {Idiom}
English Meaning: {EN_Meaning}

Example: 
{{
    "English": {Idiom_sent},
    "Plain_English": {Plain_Idiom_sent}
}}'''

############
LOCALITY_SAMPLE_ZH_EN_SYS_PROMPT = '''You are tasked with generating structured outputs for Chinese idioms. For each given idiom and its meaning in Chinese and English, follow these steps:
1. Chinese Sentence: 
    - Create 10 distinct short sentences in Chinese that fulfill all of the following constraints:
        - The sentence must not convey the meaning of the idiom.
        - The sentence must not contain any idiom, including the idiom provdided.
        - The sentence must reuse part of the visible words in the idiom (any order, any combination).
        - Each sentence must be clear, self-contained, and unambiguous.
        - Each sentence should sound natural and fluent, not stiff or mechanical.
        - Sentences must vary in subject, tone, and context to avoid repetition.
2. English Translation
     - Provide a clear, simple English translation for each sentence.

Output Format:
- Return only a valid JSON array of 10 objects, each with the following keys only:
{
  "Chinese": "<Chinese sentence>",
  "English": "<English translation>"
}'''

LOCALITY_SAMPLE_ZH_EN_USER_PROMPT = '''Chinese Idiom: {Idiom}
Chinese Meaning: {ZH_Meaning}
English Meaning: {EN_Meaning}'''

############
LOCALITY_SAMPLE_EN_EN_SYS_PROMPT ='''You are tasked with generating structured outputs for English idioms.
For each given idiom and its meaning in English, create 10 distinct short sentences in English that fulfill all of the following constraints:
- The sentence must not convey the meaning of the idiom.
- The sentence must not contain any idiom, including the idiom provdided.
- The sentence must reuse part of the visible words in the idiom (any order, any combination).
- Each sentence must be clear, self-contained, and unambiguous.
- Each sentence should sound natural and fluent, not stiff or mechanical.
- Sentences must vary in subject, tone, and context to avoid repetition.

Output Format:
- Return only a valid JSON array of 10 objects, each with the following keys only:
{
  "English": "<English sentence>"
}'''

LOCALITY_SAMPLE_EN_EN_USER_PROMPT = '''English Idiom: {Idiom}
English Meaning: {EN_Meaning}'''


############
EN_IDIOM_SPAN_SYS_PROMPT = '''Instruction:
You are given an idiom (sometimes written with blanks to indicate variable words) and a sentence that contains that idiom. Your task is to identify and output the exact span of the idiom as it appears in the sentence.

- Match idioms even if they appear in a different tense, form, or with small inserted words.

- Only output the idiom span exactly as it occurs in the sentence.

- Do not output explanations or extra text.

--
EXAMPLE 1:
Input:
Idiom: hit the ___
Sentence: After the long trip, he was so tired that he hit the bed immediately.

Output:
hit the bed

--
EXAMPLE 2:
Input:
Idiom: let the cat out of the bag
Sentence: She accidentally let the cats out of the bag during the conversation.

Output:
let the cats out of the bag

--
EXAMPLE 3:
Input:
Idiom: throw in the towel
Sentence: Realizing the competition was too strong, he finally threw in the dirty towel.

Output:
threw in the dirty towel
'''

ZH_IDIOM_SPAN_SYS_PROMPT = '''Instruction:
You are given a Chinese idiom and a Chinese sentence that contains that idiom. Your task is to identify and output the exact span of the idiom as it appears in the sentence.

- When matching, treat simplified and traditional characters as equivalent.

- Only output the idiom span exactly as it occurs in the sentence.

- Do not output explanations or extra text.

--
EXAMPLE 1:
Input:
Idiom: 随心所欲
Sentence: 他做事總是隨心所欲，從不顧及他人感受。

Output:
隨心所欲
'''


IDIOM_SPAN_USER_PROMPT ='''Input:
Idiom: {Idiom}
Sentence: {SENTENCE}

Output:'''

############
PROMPT_DICT = {
    'SENT_IDIOM_ZH_EN':
        {'sys':SENT_IDIOM_ZH_EN_SYS,
         'user':SENT_IDIOM_ZH_EN_USER},
    'SENT_IDIOM_EN_EN':
        {'sys':SENT_IDIOM_EN_EN_SYS,
         'user':SENT_IDIOM_EN_EN_USER},
    'GENERAL_TRANS':
        {'sys':GENERAL_TRANS_SYS,
         'user':GENERAL_TRANS_USER},
    'LITERAL_TRANS':
        {'sys':LITERAL_SYS_PROMPT,
         'user':LITERAL_USER_PROMPT},
    'TRANS_SCALE_EVAL':
        {'sys':SCALE_TRANS_EVAL_SYS_PROMPT,
         'user':SCALE_TRANS_EVAL_USER_PROMPT},
    'TRANS_BINARY_EVAL':
        {'sys':BINARY_TRANS_EVAL_SYS_PROMPT,
         'user':BINARY_TRANS_EVAL_USER_PROMPT},
    'GENERALITY_ZH_EN':
        {'sys':GENERALITY_SAMPLE_ZH_EN_SYS_PROMPT,
         'user':GENERALITY_SAMPLE_ZH_EN_USER_PROMPT},
    'GENERALITY_EN_EN':
        {'sys':GENERALITY_SAMPLE_EN_EN_SYS_PROMPT,
         'user':GENERALITY_SAMPLE_EN_EN_USER_PROMPT},
    'LOCALITY_ZH_EN':
        {'sys':LOCALITY_SAMPLE_ZH_EN_SYS_PROMPT,
         'user':LOCALITY_SAMPLE_ZH_EN_USER_PROMPT},
    'LOCALITY_EN_EN':
        {'sys':LOCALITY_SAMPLE_EN_EN_SYS_PROMPT,
         'user':LOCALITY_SAMPLE_EN_EN_USER_PROMPT},
    'EN_IDIOM_SPAN_IDENTIFY':
        {'sys':EN_IDIOM_SPAN_SYS_PROMPT,
         'user':IDIOM_SPAN_USER_PROMPT},
    'ZH_IDIOM_SPAN_IDENTIFY':
        {'sys':ZH_IDIOM_SPAN_SYS_PROMPT,
         'user':IDIOM_SPAN_USER_PROMPT},
}

