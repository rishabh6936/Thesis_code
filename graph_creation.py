import networkx as nx
import spacy
import matplotlib.pyplot as plt
import hashlib
from networkx.classes import graph
from trie_conceptnet import Trie
from trie_structure import Trie_hash
from Graph_builder import GraphBuilder
from knowledge_extractor import KnowExtract
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import re
from tqdm import tqdm

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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


def edge_creation(graph, dictionary, gb, new_nodes, trie_hash):
    email = [value for key, value in new_nodes if key == 'Text']
    if isinstance(email, list):
        email = ''.join(email)  # Convert list to string
    email = preorocess_mail(email)

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
        norm_entity = normalize_nodes(entity)
        hash_value = hashing_function(norm_entity)
        node_check = trie_hash.query(hash_value)
        if node_check:  # if entity already in graph
            graph.add_edge(email, norm_entity, edge_type='belongs_to')
        else:  # if entity not in graph
            spell_check = gb.trie.query(norm_entity)
            if spell_check != ([], []):  # node not misspelled, conceptNet returns something
                set_trie_hash(trie_hash, hash_value, norm_entity)  # add to hash
                graph.add_edge(email, norm_entity, edge_type='belongs_to')  # add to graph
            else:  # nodes probably misspelled, conceptNet returns nothing
                similar_words = gb.trie.search(entity.lower(), 1)  # fetch similar words
                if similar_words != []:
                 pick_best_match(entity,similar_words, email,gb)

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


def pick_best_match(entity, similar_words, email,gb):
    sentence_pattern = r'([^.?!]*[.?!])'
    sentences = re.findall(sentence_pattern, email)
    word_similarity_data = []
    # Iterate through sentences and check if the entity is present
#    word = word[0].split('/')[3]

    entity_pattern = re.compile(rf'\b{re.escape(entity)}\b')

    for sentence in sentences:
        if re.search(entity_pattern, sentence):
            sentence_hit = sentence.strip()
            break  # If you want to stop after the first match

    # Find the token index corresponding to the entity
    misspelled_entity_embedding= find_token_indices( entity,sentence_hit)
    for word in similar_words:
        x, similar_word_embedding = gb.trie.query(word)
        score = compute_similarity(misspelled_entity_embedding,similar_word_embedding)
        word_similarity_data.append([word, score])

    return word_similarity_data




def compute_similarity(ms_entity_embedding,similar_word_embedding):
    similarity = model.similarity(ms_entity_embedding, similar_word_embedding)
    return similarity


def preorocess_mail(email):
    text = "".join([s for s in email.splitlines(True) if s.strip("\r\n")])
    return text



def find_token_indices( entity, sentence_hit):
    """Find the token indices corresponding to the entity word."""
    split_sentence= sentence_hit.split()
    if entity in split_sentence:
        entity_id = split_sentence.index(entity)
    else: return

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
    nx.draw_networkx(graph, with_labels=True)  # Adjust font_size here
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
