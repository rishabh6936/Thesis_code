import torch
from graph_creation import node_creation, visualise_graph, save_graph_pickle, load_graph_pickle
import networkx as nx
import json
import matplotlib.pyplot as plt
import hashlib
from networkx.classes import graph
from trie_conceptnet import Trie
from trie_structure import Trie_hash
from Graph_builder import GraphBuilder
"""import spacy
from challenging_negatives_training import create_corrupted_graphs, preprocess_graphs
import pickle
from sentence_transformers import SentenceTransformer
import gc
import fasttext
import fasttext.util
"""

def main():
    dictionary = json.load(open('/Users/rishabhsingh/Downloads/tumail.json', 'r'))
    trie_hash = Trie_hash()
    trie = Trie()
    gb = GraphBuilder()
    graph = nx.Graph()
    node_creation(graph, trie_hash, dictionary,gb)
    visualise_graph(graph)
#    load_graph_pickle()
    save_graph_pickle(graph)

"""
def main():
    # Download and load the fasttext model
    fasttext.util.download_model('de', if_exists='ignore')  # German model
    ft = fasttext.load_model('cc.de.300.bin')

    # gb = GraphBuilder()
    # Load pre-trained word vectors (e.g., 'glove-wiki-gigaword-100' or 'word2vec-google-news-300')
    # glove_vectors = gensim.downloader.load('word2vec-google-news-300')  # or 'word2vec-google-news-300'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model.to(torch.device('cpu'))

    save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph_500.pkl'
    with open(save_path, mode='rb') as f:
        graph = pickle.load(f)
    email_nodes = [node for node, attr in graph.nodes(data=True) if attr.get('node_type') == 'email']
    typo_graph, similar_word_graph = create_corrupted_graphs(graph,email_nodes)
    with open('/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/typo_graph_500.pkl', 'wb') as f:
        pickle.dump(typo_graph ,f)
    with open('/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/similar_word_graph_500.pkl', 'wb') as f:
        pickle.dump(similar_word_graph ,f)
    preprocess_graphs(typo_graph, similar_word_graph, graph)
"""


if __name__ == '__main__':
    main()