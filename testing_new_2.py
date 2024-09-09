from Graph_builder import GraphBuilder
from knowledge_extractor import KnowExtract
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import re
from tqdm import tqdm
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = model.tokenizer




def find_token_indices( entity, sentence_hit):
    """Find the token indices corresponding to the entity word."""
    split_sentence = sentence_hit.split()
    sentence = ' '.join(sentence_hit.split())
    entity_pattern = re.compile(rf'\b{re.escape(entity)}\b')
    match = re.search(entity_pattern, sentence)
    char_count = 0
    if match:
        # Get the matched entity
        entity = match.group(0)

        # Check if the entity exists in the split sentence and get its index
        for word in split_sentence:
            char_count += len(word) + 1  # +1 for the space or separator after each word
            if char_count >= match.regs[0][0]:
                entity_id = split_sentence.index(word)
                break
    else:
            print("Entity not found in split sentence")
    embeddings = model.encode(sentence_hit, output_value="token_embeddings")
    embeddings = embeddings[1:-1]              #remove [CLS] and [SEP]


    enc = tokenizer(sentence_hit, add_special_tokens=False)

    word_ids_arr = []
    # BatchEncoding.word_ids returns a list mapping words to tokens
    for w_idx in set(enc.word_ids()):
        # BatchEncoding.word_to_tokens tells us which and how many tokens are used for the specific word
        start, end = enc.word_to_tokens(w_idx)
        word_ids_arr.append(list(range(start, end)))

    entity_embeddings = embeddings[word_ids_arr[entity_id]]
    sum_entity_embeddings = entity_embeddings.sum(dim=0, keepdim=True)

    return sum_entity_embeddings

def compute_similarity(ms_entity_embedding,similar_word_embedding,word):
    # Ensure the tensors are on the CPU
    ms_entity_embedding_cpu = ms_entity_embedding.cpu().detach().numpy()
    similar_word_embedding_cpu = similar_word_embedding.cpu().detach().numpy()

    similarity = cosine_similarity(ms_entity_embedding_cpu, similar_word_embedding_cpu)
    return similarity

def pick_best_match(entity, similar_words, email,gb):
    sentence_pattern = r'([^.?!]*[.?!])'
    sentences = re.findall(sentence_pattern, email)
    word_similarity_data = []
    sentence_hit = ''
    # Iterate through sentences and check if the entity is present
#    word = word[0].split('/')[3]

    entity_pattern = re.compile(rf'\b{re.escape(entity)}\b')

    for sentence in sentences:
        if re.search(entity_pattern, sentence):
            sentence_hit = sentence.strip()
            break  # If you want to stop after the first match

    # Find the token index corresponding to the entity
    if sentence_hit != '':
        misspelled_entity_embedding= find_token_indices( entity,email)
        for word in similar_words:
            x, similar_word_embedding = gb.trie.query((word[0].split('/'))[3])
            similar_word_embedding_tensors = [torch.from_numpy(embedding) for embedding in similar_word_embedding]
            stacked_embeddings = torch.stack(similar_word_embedding_tensors)
            summed_embeddings = stacked_embeddings.sum(dim=0)
            s_word_emb = summed_embeddings.unsqueeze(0)
            score = compute_similarity(misspelled_entity_embedding,s_word_emb, word)
            word_similarity_data.append([word, score])
        print('Misspelled entity:', entity)
        print(word_similarity_data)
        return word_similarity_data
    else:
        return None

def main():
    email = 'Auf unserer Website finden Sie detaillierte Beschreibungen und weitere Informationen zur Bearbeitung von Modilen und der Eintragung von Prüfer*innen https://www.tu.berlin/go209536/. Welche Personengruppen als Prüfer*innen bestellt werden können, entnehmen Sie bitte der Handreichung im Anhang.'
    entity = 'Modilen'
    gb = GraphBuilder()
    similar_words = gb.trie.search(entity.lower(), 1)  # fetch similar words
    if similar_words != []:
        pick_best_match(entity, similar_words, email, gb)

if __name__ == '__main__':
    main()
