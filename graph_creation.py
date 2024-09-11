import networkx as nx
import spacy
import matplotlib.pyplot as plt
import hashlib
import pickle
from nltk import RegexpTokenizer
from Graph_builder import GraphBuilder
from knowledge_extractor import KnowExtract
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import re
from tqdm import tqdm
import gc

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.to(torch.device('cpu'))
tokenizer = model.tokenizer

nlp = spacy.load('de_core_news_sm')
path = '/Users/rishabhsingh/Shira_thesis/Crawlers/dia_trans_net/data/conceptNet_embs',
save_path = '/Users/rishabhsingh/PycharmProjects/Mails_Graph/datasets/',
emb_file_name = 'conceptnet_embs_eng'


def node_creation(graph, trie_hash, dictionary, gb):
    for records in tqdm(dictionary, total=len(dictionary), desc="Processing records"):
        new_nodes = []
        for key, value in records.items():
            if value is not None:
                hash_value = hashing_function(value)
                x = trie_hash.query(hash_value)  # if the node not present in the Graph
                if x == []:
                    #                    graph.add_node(value, node_type=key, unique_id=hash_value)     #add node to the graph
                    set_trie_hash(trie_hash, hash_value, value)  # add node to the Trie
                    new_nodes.append([key, value])  # make a list of nodes to make edges with
                if x != []:
                    new_nodes.append([key, value])
        edge_creation(graph, dictionary, gb, new_nodes, trie_hash)  # send the new nodes for edge creation
        gc.collect()


def edge_creation(graph, dictionary, gb, new_nodes, trie_hash):
    best_match_word = ''
    email = [value for key, value in new_nodes if key == 'Text']
    if isinstance(email, list):
        email = ''.join(email)  # Convert list to string
    email = preprocess_mail(email)

    msg_sub = KnowExtract(email, gb.trie, 2)
    for hop in range(4):
        ex_nodes = msg_sub.new_hop(hop_nr=hop, k=10)
        if ex_nodes == 0:
            break

    """First add a directed edge from Email to all its respective related nodes, these nodes are not normalised"""
    for key, value in new_nodes:
        if key != "Text":
            graph.add_edge(email, value, edge_type=key)

    for entity in msg_sub.data['knowledge_nodes']:
        std_entity = entity.replace('ß', 'ss')
        norm_entity = normalize_nodes(entity)
        hash_value = hashing_function(norm_entity)
        node_check = trie_hash.query(hash_value)
        if node_check:  # if entity already in graph
            graph.add_edge(email, norm_entity, edge_type='belongs_to')
        else:  # if entity not in graph
            spell_check = gb.trie.query(norm_entity.lower())
            if spell_check != ([], []):  # node not misspelled, conceptNet returns something
                set_trie_hash(trie_hash, hash_value, norm_entity)  # add to hash
                graph.add_edge(email, norm_entity, edge_type='belongs_to')  # add to graph
            else:  # nodes probably misspelled, conceptNet returns nothing
                similar_words = gb.trie.search(std_entity.lower(), 1)  # fetch similar words
                if similar_words != []:
                    similarity_array = pick_best_match(entity, similar_words, email, gb)
                    best_match = get_max_similarity_word(similarity_array)
                    if best_match == '' or best_match is None:
                        continue
                    best_match_word = split_conceptnet_word(best_match[0])
                    norm_entity = normalize_nodes(best_match_word)
                    set_trie_hash(trie_hash, hash_value, norm_entity)
                    graph.add_edge(email, norm_entity, edge_type='belongs_to')

    for entity in msg_sub.data['edges_before']:
        for i in range(len(entity[0])):
            graph.add_edge(entity[0][i], entity[1], edge_type=entity[2])

    for entity in msg_sub.data['edges_after']:
        for i in range(len(entity[0])):
            graph.add_edge(entity[0][i], entity[1], edge_type=entity[2])

    """for entity in msg_sub.graph_edges:
        split_word = entity[0].split('/')
        actual_word = split_word[3]
        lowercase_entity = np.char.lower(msg_sub.entities)
        occurrence_nodes_before = msg_sub.data['occurence_nodes_before']
        occurrence_nodes_after = msg_sub.data['occurence_nodes_after']

        if actual_word in lowercase_entity:
            positions = np.where(actual_word == lowercase_entity)[0]
        if positions.size > 0:
            position = positions[0]
            node_before =occurrence_nodes_before[position]
            node_after = occurrence_nodes_after[position]

            if node_before != 'Empty nodes' and node_after != 'Empty nodes':
                graph.add_edge(node_before, entity[0], edge_type='mentioned_in' )
                graph.add_edge( entity[0],node_after, edge_type='comes_before')
                graph.add_edge(email, node_before, edge_type='belongs_to')
                graph.add_edge(email, node_after, edge_type='belongs_to')"""


def split_conceptnet_word(conceptnet_word):
    split_word = conceptnet_word[0].split('/')
    actual_word = split_word[3]
    return actual_word


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
        for canditate_word in similar_words:
            replaced_sentence = replace_misspelled_candidate(misspelled_word,split_conceptnet_word(canditate_word),sentence_hit)
            if replaced_sentence == []:
                return None
            candidate_embedding = find_token_indices_embed(split_conceptnet_word(canditate_word), replaced_sentence)
            c_trie_emb = get_trie_embedding(gb,canditate_word)
            score = compute_similarity(candidate_embedding,c_trie_emb)
            word_similarity_data.append([canditate_word, score])
#        print('Misspelled entity:', misspelled_word)
#        print(word_similarity_data)
        return word_similarity_data
    else:
        return None

def get_trie_embedding(gb,word):
    x, candidate_trie_emb = gb.trie.query(split_conceptnet_word(word))
    candidate_trie_emb_tensors = [torch.from_numpy(embedding) for embedding in candidate_trie_emb]
    stacked_embeddings = torch.stack(candidate_trie_emb_tensors)
    summed_embeddings = stacked_embeddings.sum(dim=0)
    c_word_emb = summed_embeddings.unsqueeze(0)
    return c_word_emb
def replace_misspelled_candidate(misspelled_word, similar_word, sentence):
    tokenizer = RegexpTokenizer(r'\d+,\d+[a-zA-Z]|\w+')
    split_sentence = tokenizer.tokenize(sentence)

#    split_sentence = sentence.split()
    replaced_sentence = []
    for word in split_sentence:
        if word == misspelled_word:
            replaced_sentence.append(similar_word)
        else:
            replaced_sentence.append(word)

    replaced_sentence = ' '.join(replaced_sentence)
    return replaced_sentence



def get_max_similarity_word(word_similarity_data):
    if not word_similarity_data:
        return None  # Handle the case if the list is empty
    # Initialize max values
    max_word = ''
    max_score = float('-inf')

    # Iterate through the word_similarity_data list
    for word, score in word_similarity_data:
        if score > max_score:
            max_word = word
            max_score = score
    return max_word, max_score


def compute_similarity(ms_entity_embedding, can_trie_embedding):
    # Ensure the tensors are on the CPU
    ms_entity_embedding_cpu = ms_entity_embedding.cpu().detach().numpy()
    similar_word_embedding_cpu = can_trie_embedding.cpu().detach().numpy()

    similarity = cosine_similarity(ms_entity_embedding_cpu, similar_word_embedding_cpu)
    return similarity


def preprocess_mail(email):
    text = "".join([s for s in email.splitlines(True) if s.strip("\r\n")])
    return text


def find_token_indices_embed(entity, sentence_hit):
    """Find the token indices corresponding to the entity word."""
    split_sentence = sentence_hit.split()
    sentence = ' '.join(sentence_hit.split())
    entity_pattern = re.compile(rf'\b{re.escape(entity)}\b')
    match = re.search(entity_pattern, sentence.lower())
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
    embeddings = embeddings[1:-1]  # remove [CLS] and [SEP]

    enc = tokenizer(sentence_hit, add_special_tokens=False)

    word_ids_arr = []
    # BatchEncoding.word_ids returns a list mapping words to tokens
    for w_idx in set(enc.word_ids()):
        # BatchEncoding.word_to_tokens tells us which and how many tokens are used for the specific word
        start, end = enc.word_to_tokens(w_idx)
        word_ids_arr.append(list(range(start, end)))

    entity_embeddings = embeddings[word_ids_arr[entity_id]]
    sum_entity_embeddings = entity_embeddings.sum(dim=0, keepdim=True)
    del embeddings

    return sum_entity_embeddings


#    entity_embedding = embeddings[entity_tokens[]]


def set_trie_hash(trie_hash, hash_value, value):
    trie_hash.insert_dataset(hash_value, value)


def hashing_function(value):  # hashing to asign a unique Id to each node
    hash_object = hashlib.sha256()
    hash_object.update(value.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    return hash_digest


def normalize_nodes(entity):
    filtered_entities = []
    doc = nlp(entity)
    for token in doc:
        lemma = token.lemma_
        normalised_entity = lemma

    return normalised_entity


def visualise_graph(graph):
    plt.figure(figsize=(10, 7))  # Optional: Adjust the figure size for better visualization
    nx.draw_networkx(graph, with_labels=False)  # Adjust font_size here
    plt.show()


def save_graph(graph):
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, np.str_):
                graph.nodes[node][key] = str(value)

    if isinstance(graph, nx.MultiGraph):
        for u, v, k, data in graph.edges(data=True, keys=True):
            for key, value in data.items():
                if isinstance(value, np.str_):
                    graph.edges[u, v, k][key] = str(value)

    for idx, (u, v, key, data) in enumerate(graph.edges(keys=True, data=True)):
        data['id'] = idx

    nx.write_graphml(graph, "/Users/rishabhsingh/PycharmProjects/Mails_Graph/datasets/graph.graphml")

def save_graph_pickle(graph):
    with open('/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph.pkl', 'wb') as f:
        pickle.dump(graph, f)