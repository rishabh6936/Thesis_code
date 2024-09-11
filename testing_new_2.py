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




def find_token_indices_embed( entity, sentence_hit):
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
            if char_count > match.regs[0][0]:
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

def replace_misspelled_candidate(misspelled_word, similar_word, sentence):
    split_sentence = sentence.split()
    replaced_sentence = []
    for word in split_sentence:
        if word == misspelled_word:
            replaced_sentence.append(similar_word)
        else:
            replaced_sentence.append(word)

    replaced_sentence = ' '.join(replaced_sentence)
    return replaced_sentence



def compute_similarity(ms_entity_embedding,similar_word_embedding,word):
    # Ensure the tensors are on the CPU
    ms_entity_embedding_cpu = ms_entity_embedding.cpu().detach().numpy()
    similar_word_embedding_cpu = similar_word_embedding.cpu().detach().numpy()

    similarity = cosine_similarity(ms_entity_embedding_cpu, similar_word_embedding_cpu)
    return similarity

def pick_best_match(misspelled_word, similar_words, email,gb):
    sentence_pattern = r'([^.?!]*[.?!])'
    sentences = re.findall(sentence_pattern, email)
    word_similarity_data = []
    sentence_hit = ''
    # Iterate through sentences and check if the entity is present
#    word = word[0].split('/')[3]

    entity_pattern = re.compile(rf'\b{re.escape(misspelled_word)}\b')

    for sentence in sentences:
        if re.search(entity_pattern, sentence):
            sentence_hit = sentence.strip()
            break  # If you want to stop after the first match

    # Find the token index corresponding to the entity
    if sentence_hit != '':
        misspelled_entity_embedding= find_token_indices_embed( misspelled_word,email)
        for canditate_word in similar_words:
            replaced_sentence = replace_misspelled_candidate(misspelled_word,(canditate_word[0].split('/'))[3],sentence_hit)
            candidate_embedding = find_token_indices_embed((canditate_word[0].split('/'))[3], replaced_sentence)
            """x, candidate_trie_emb = gb.trie.query((word[0].split('/'))[3])
            candidate_trie_emb_tensors = [torch.from_numpy(embedding) for embedding in candidate_trie_emb]
            stacked_embeddings = torch.stack(candidate_trie_emb_tensors)
            summed_embeddings = stacked_embeddings.sum(dim=0)
            c_word_emb = summed_embeddings.unsqueeze(0)"""
            c_trie_emb = get_trie_embedding(gb,canditate_word)
            score = compute_similarity(candidate_embedding,c_trie_emb, canditate_word)
            word_similarity_data.append([canditate_word, score])
        print('Misspelled entity:', misspelled_word)
        print(word_similarity_data)
        return word_similarity_data
    else:
        return None
def get_trie_embedding(gb,word):
    x, candidate_trie_emb = gb.trie.query((word[0].split('/'))[3])
    candidate_trie_emb_tensors = [torch.from_numpy(embedding) for embedding in candidate_trie_emb]
    stacked_embeddings = torch.stack(candidate_trie_emb_tensors)
    summed_embeddings = stacked_embeddings.sum(dim=0)
    c_word_emb = summed_embeddings.unsqueeze(0)
    return c_word_emb


def main():
    email = 'In der Veranstaltung ""Introprog"" dieses Wintersemesters erzielte ich bei den Aufgeben 44 von 50 möglichen Punkten und im Abschlusstest 35,8 Punkte. .'
    entity = 'Aufgeben'
    gb = GraphBuilder()
#    replace_misspelled_candidate('Modilen', 'Modulen', email)
    similar_words = gb.trie.search(entity.lower(), 1)  # fetch similar words
    if similar_words != []:
        pick_best_match(entity, similar_words, email, gb)

if __name__ == '__main__':
    main()
