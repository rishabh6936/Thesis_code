import numpy as np
import faiss
from gensim.parsing.preprocessing import STOPWORDS
from sentence_transformers import SentenceTransformer
from nltk import RegexpTokenizer
import re
import torch
import spacy

text = 'Ich möchte meine Klaszr Punkte wissen, Wie ist meine Note eigentlich? Aus welche Fachgebeet sind Sie. Das ist suuper'
german_pronouns = set(['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'der', 'die', 'das'])  # German equivalents
english_pronouns = set(['I', 'you', 'he', 'she', 'it', 'we', 'they'])

# Set of integers from 0 to 9 as strings
integer_stopwords = set([str(i) for i in range(1000)])

# Combine all stopwords
stopwords = STOPWORDS.union(english_pronouns).union(german_pronouns).union(integer_stopwords)




tokenizer = RegexpTokenizer(r"\w+")
text_tokenized = tokenizer.tokenize(text)
entities = [word for word in text_tokenized if not word.lower() in stopwords]
print(entities)