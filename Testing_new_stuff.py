from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = model.tokenizer

str = "Sehr geehrter Herr Hotait, die Gesamtpunktzahl von 79,83 in ISIS ist bereits die resultierende nach der Klausureinsicht"

enc = tokenizer(str, add_special_tokens=False)

desired_output = []
tokens = tokenizer.tokenize(str)
#BatchEncoding.word_ids returns a list mapping words to tokens
for w_idx in set(enc.word_ids()):
    #BatchEncoding.word_to_tokens tells us which and how many tokens are used for the specific word
    start, end = enc.word_to_tokens(w_idx)
    desired_output.append(list(range(start,end)))

print(desired_output)
print(tokens)