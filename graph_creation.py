import networkx as nx
import spacy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import hashlib
import pickle
import nltk
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
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
from hgt_link_prediction import LinkPrediction

nltk.download('punkt_tab', quiet=True)

from torch_geometric.utils.convert import to_networkx, from_networkx

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.to(torch.device('cpu'))
tokenizer = model.tokenizer

nlp_de = spacy.load('de_core_news_sm')
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
    link_prediction_mechanism(graph)


def link_prediction_mechanism(graph):
    edges_to_remove = []
    nodes_to_remove = []
    edges_to_consider = []
    #Get predictions for each node
    predicted_labels = LinkPrediction(graph).predicted_labels
    # assign each edge an attribute predicted_label
    #required_edges = get_edges_of_type("email", "belongs_to", "noun", graph)
    i = 0
    for u, v in graph.edges():
        if graph.edges[u, v].get('edge_type') == 'mentions' and graph.nodes[u].get('node_type') == "sentence" and graph.nodes[v].get('node_type') == 'noun':
            edges_to_consider.append((u, v))
            graph.edges[u, v]['predicted_label'] = predicted_labels[i] # Adding new attribute
            i = i+1

    #delete the noun nodes with attributes is_predicted = True and predicted_label = 0
    for u, v in graph.edges():
        if graph.edges[u, v].get('edge_type') == 'mentions' and graph.nodes[u].get('node_type') == "sentence" and graph.nodes[v].get('node_type') == 'noun':
            if graph.edges[u, v]['predicted_label'] == 0.0 and graph.edges[u, v]['is_predicted'] == "True":
                edges_to_remove.append((u, v))
                nodes_to_remove.append(v)

    graph.remove_edges_from(edges_to_remove)

    for node in nodes_to_remove:
        if graph.degree(node) == 0:  # Check if node is no longer connected
            graph.remove_node(node)



    print(predicted_labels)


def get_edges_of_type(src_type, edge_type, tgt_type, graph):
        edges = []
        # Iterate through all edges and filter based on node types and edge types
        for u, v, data in graph.edges(data=True):
            # Check if the edge type and node types match
            if data.get('edge_type') == edge_type and \
                    graph.nodes[u].get('node_type') == src_type and \
                    graph.nodes[v].get('node_type') == tgt_type:
                edges.append({
                    'source': u,
                    'target': v,
                    'edge_type': data["edge_type"],
                    'is_predicted': data["is_predicted"]
                })

        return edges
def edge_creation(graph, dictionary, gb, new_nodes, trie_hash):
    best_match_word = ''
    email = [value for key, value in new_nodes if key == 'Text']
    if isinstance(email, list):
        email = ''.join(email)  # Convert list to string
    email = preprocess_mail(email)
    graph.add_node(email, node_type='email', embedding=get_node_embedding(email))

    sentences = sent_tokenize(email)

    prev_sentence_node = None  # To track the previous sentence node for "comes after" edges
    for i, sentence in enumerate(sentences):
        # Create a unique node identifier for the sentence
        # Add sentence node
        graph.add_node(sentence, node_type='sentence', embedding=get_node_embedding(sentence))

        # Add "child of" edge between email and sentence
        graph.add_edge(email, sentence, edge_type='child_of')
        graph.add_edge(sentence, email, edge_type='father_of')

        # Add "comes after" edge between consecutive sentences
        if prev_sentence_node:
            graph.add_edge(prev_sentence_node, sentence, edge_type='comes_after')

        # Update previous sentence node
        prev_sentence_node = sentence

    msg_sub = KnowExtract(email, gb.trie, 2)
    for hop in range(2):
        ex_nodes = msg_sub.new_hop(hop_nr=hop, k=100)
        if ex_nodes == 0:
            break

    """First add a directed edge from Email to all its respective related nodes, these nodes are not normalised"""
    for key, value in new_nodes:
        if key != "Text":
            node_type = get_node_type(key)
            graph.add_node(value, node_type=node_type, embedding=get_node_embedding(value))
            graph.add_edge(email, value, edge_type=get_clean_edge_type(key))

    for entity in msg_sub.data['knowledge_nodes']:
        std_entity = entity.replace('ß', 'ss')
        norm_entity = normalize_nodes(entity)
        hash_value = hashing_function(norm_entity)
        node_check = trie_hash.query(hash_value)
        if node_check:  # if entity already in graph
            graph.add_node(norm_entity, node_type='noun', embedding=get_node_embedding(norm_entity))
            sentences = get_corresponding_sentence(email,entity)
            for sentence in sentences:
               graph.add_edge(sentence, norm_entity, edge_type='mentions', is_predicted=False)
            if msg_sub.graph_edges is not None:
                add_context_nodes(graph, norm_entity, msg_sub.graph_edges)
        else:  # if entity not in graph
            spell_check = gb.trie.query(norm_entity.lower())
            if spell_check != ([], []):  # node not misspelled, conceptNet returns something
                set_trie_hash(trie_hash, hash_value, norm_entity)  # add to hash
                graph.add_node(norm_entity, node_type='noun', embedding=get_node_embedding(norm_entity))
                sentences = get_corresponding_sentence(email, entity)
                for sentence in sentences:
                    graph.add_edge(sentence, norm_entity, edge_type='mentions', is_predicted=False)
                if msg_sub.graph_edges is not None:
                    add_context_nodes(graph, norm_entity, msg_sub.graph_edges)
            else:  # nodes probably misspelled, conceptNet returns nothing
                similar_words = gb.trie.search(std_entity.lower(), 1)  # fetch similar words
                #version 1 with embedding similarity
                """if similar_words != []:
                    similarity_array = pick_best_match(entity, similar_words, email, gb)
                    best_match = get_max_similarity_word(similarity_array)
                    if best_match == '' or best_match is None:
                        continue
                    best_match_word = split_conceptnet_word(best_match[0])
                    norm_entity = normalize_nodes(best_match_word)
                    set_trie_hash(trie_hash, hash_value, norm_entity)
                    graph.add_node(norm_entity, node_type='noun', embedding=get_node_embedding(norm_entity))
                    graph.add_edge(email, norm_entity, edge_type='belongs_to', is_predicted=True)
                    if msg_sub.graph_edges is not None:
                       add_context_nodes(graph, norm_entity, msg_sub.graph_edges)"""

                #version 2 with the trained model
                if similar_words != []:
                    #similarity_array = pick_best_match(entity, similar_words, email, gb)
                    #best_match = get_max_similarity_word(similarity_array)
                    #if best_match == '' or best_match is None:
                    #    continue
                    for word in similar_words[:5]:
                        best_match_word = split_conceptnet_word(word)
                        norm_entity = normalize_nodes(best_match_word)
                        set_trie_hash(trie_hash, hash_value, norm_entity)
                        sentences = get_corresponding_sentence(email, entity)
                        for sentence in sentences:
                            graph.add_edge(sentence, norm_entity, edge_type='mentions', is_predicted=True)
                        if msg_sub.graph_edges is not None:
                            add_context_nodes(graph, norm_entity, msg_sub.graph_edges)

                #version 3, to train HGT with, without any misspelling mechanism
                """set_trie_hash(trie_hash, hash_value, norm_entity)  # add to hash
                graph.add_node(norm_entity, node_type='noun', embedding=get_node_embedding(norm_entity))
                sentences = get_corresponding_sentence(email, entity)
                for sentence in sentences:
                    graph.add_edge(sentence, norm_entity, edge_type='mentions', is_predicted=True)
                if msg_sub.graph_edges is not None:
                    add_context_nodes(graph, norm_entity, msg_sub.graph_edges)"""



    for entity in msg_sub.data['edges_before']:
        for i in range(len(entity[0])):
            if entity[0][i] != 'Empty_node':
                if not graph.has_node(entity[0][i]):
                   graph.add_node(entity[0][i], node_type='substring', embedding=get_node_embedding(entity[0][i]))
                if not graph.has_node(entity[1]):
                   graph.add_node(entity[1], node_type='noun', embedding=get_node_embedding(entity[1]))
                graph.add_edge(entity[0][i], entity[1], edge_type=str(entity[2][0]))
                graph.add_edge(entity[1], entity[0][i], edge_type='comes_after')

    for entity in msg_sub.data['edges_after']:
        for i in range(len(entity[0])):
            if entity[0][i] != 'Empty_node':
                if not graph.has_node(entity[0][i]):
                    graph.add_node(entity[0][i], node_type='substring', embedding=get_node_embedding(entity[0][i]))
                if not graph.has_node(entity[1]):
                    graph.add_node(entity[1], node_type='noun', embedding=get_node_embedding(entity[1]))
                graph.add_edge(entity[0][i], entity[1], edge_type=str(entity[2][0]))
                graph.add_edge(entity[1], entity[0][i], edge_type='comes_before')

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

def get_clean_edge_type(key):
    if key == 'Von: (Adresse)':
        return 'sent_from'
    if key == 'An: (Adresse)':
        return 'sent_to'
    if key == 'Betreff':
        return 'title'
    if key == 'An: (Name)':
        return 'to_person'
    if key == 'Von: (Name)':
        return 'from_person'




def get_node_embedding(node):
    embeddings = model.encode(node)
    return embeddings


def get_node_type(key):
    if key == 'Betreff':
        return 'betreff'
    # Regex pattern
    pattern = r'\(([^)]+)\)'
    # Find all matches
    matches = re.findall(pattern, key)
    return matches[0].lower()


def add_context_nodes(graph, norm_entity, graph_edges):
    for edges in graph_edges:
        word = split_conceptnet_word(edges)
        norm_word = normalize_nodes(word)
        if norm_word.lower() == norm_entity.lower():
            graph.add_node(edges[1], node_type='context', embedding=get_node_embedding(edges[1]))
#            graph.add_edge(norm_entity, edges[1], edge_type=edges[2])
            graph.add_edge(norm_entity, edges[1], edge_type='has_context')

def split_conceptnet_word(conceptnet_word):
    split_word = conceptnet_word[0].split('/')
    actual_word = split_word[3]
    return actual_word

def get_corresponding_sentence(email, entity):
    # Tokenize the email into sentences
    sentences = sent_tokenize(email)

    # Compile a regex pattern to match the entity as a whole word
    entity_pattern = re.compile(rf'\b{re.escape(entity)}\b', re.IGNORECASE)

    # Collect all sentences containing the entity
    matching_sentences = [
        sentence.strip() for sentence in sentences if re.search(entity_pattern, sentence)
    ]
    return matching_sentences



def pick_best_match(misspelled_word, similar_words, email,gb):
    sentence_pattern = r'([^.?!]*[.?!])'
    sentences = re.findall(sentence_pattern, email)
    word_similarity_data = []
    sentence_hit = ''
    entity_pattern = re.compile(rf'\b{re.escape(misspelled_word)}\b')

    for sentence in sentences:
        if re.search(entity_pattern, sentence):
            sentence_hit = sentence.strip()
            break  # If you want to stop after the first match

    # Find the token index corresponding to the entity
    if sentence_hit != '':
        for canditate_word in similar_words:
            replaced_sentence = replace_misspelled_candidate(misspelled_word,split_conceptnet_word(canditate_word),sentence_hit)
            candidate_embedding = find_token_indices_embed(split_conceptnet_word(canditate_word), replaced_sentence)
            if replaced_sentence == [] or candidate_embedding is None:
                return None

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
    if text =='':
        text = 'Empty Body'
    return text


def find_token_indices_embed(entity, sentence_hit):
    """Find the token indices corresponding to the entity word."""
#    split_sentence = sentence_hit.split()
#    sentence = ' '.join(sentence_hit.split())
    tokenizer_reg = RegexpTokenizer(r'\d+,\d+[a-zA-Z]|\w+')
    split_sentence = tokenizer_reg.tokenize(sentence_hit)
    sentence = ' '.join(split_sentence)
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
        return None
    embeddings = model.encode(sentence, output_value="token_embeddings")
    embeddings = embeddings[1:-1]  # remove [CLS] and [SEP]

    enc = tokenizer(sentence, add_special_tokens=False)

    word_ids_arr = []
    # BatchEncoding.word_ids returns a list mapping words to tokens
    for w_idx in set(enc.word_ids()):
        # BatchEncoding.word_to_tokens tells us which and how many tokens are used for the specific word
        start, end = enc.word_to_tokens(w_idx)
        word_ids_arr.append(list(range(start, end)))

    if max(word_ids_arr[entity_id]) >= embeddings.shape[0]:       #really weird error where tokenised array is bigger than embedding array,cheap fix
        return None

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
    normalized_entity = ''
    doc = nlp_de(entity)
    for token in doc:
        if token.ent_type_ != 'PER' and token.ent_type_ != 'ORG' and token.ent_type_ != 'GPE':
            lemma = token.lemma_
            normalized_entity = lemma
        else:
            normalized_entity = entity
    return normalized_entity


def visualise_graph(graph):
    plt.figure(figsize=(30, 35))  # Adjust the figure size for better visualization
    nx.draw_networkx(graph, with_labels=True)
    #plt.show()
    plt.savefig('saved_data/graph.png')

def visualise_graph_w_l(graph):
    plt.figure(figsize=(30, 35))  # Adjust the figure size for better visualization

    # Generate node positions using a layout algorithm (e.g., spring layout)
    pos = nx.spring_layout(graph, seed=42)  # You can choose a different layout if desired

    # Draw nodes with labels
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_labels(graph, pos, font_size=12)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, edge_color='gray')

    # Extract edge labels (edge_type) and display them
    edge_labels = nx.get_edge_attributes(graph, 'edge_type')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    # Save the plot to a file
    plt.savefig('saved_data/graph.png')
    plt.close()

def visualise_subgraph(graph):
    # Find the specific node (e.g., node 1)
    specific_node = "Susann"

    # Get all nodes connected to this specific node
    connected_nodes = list(graph.neighbors(specific_node))
    subgraph_nodes = [specific_node] + connected_nodes
    subgraph = graph.subgraph(subgraph_nodes)

    # Create a new figure
    plt.figure(figsize=(6, 6))  # Explicitly creating a new figure

    nx.draw_networkx(subgraph, with_labels=True)

    plt.title(f"Node {specific_node} and its connected nodes")
    plt.savefig('saved_data/subgraph.png')  # Save plot to a file instead of displaying





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
    with open('/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph_tf_small.pkl', 'wb') as f:
        pickle.dump(graph, f)

def load_graph_pickle():
    with open('/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph_tf_small.pkl', 'rb') as f:
        graph = pickle.load(f)
    return graph
