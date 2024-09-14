from sentence_transformers import SentenceTransformer
import torch
import spacy

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = model.tokenizer

sentence = 'Organization Committee General Chairs Wei Xiang La Trobe University Australia Carla Fabiana Chiasserini Politecnico di Torino Italy TPC y Forum Chair Henry legung University of Calgary Workshop Chairs Teng Joon Lim University of Sydney Australia Jia Hu University of Exter U'

embeddings = model.encode(sentence, output_value="token_embeddings")
#embeddings = embeddings[1:-1]  # remove [CLS] and [SEP]

enc = tokenizer(sentence, add_special_tokens=True)
print(sentence)
