import torch
from graph_creation import node_creation, visualise_graph, save_graph_pickle
import networkx as nx
import json
import matplotlib.pyplot as plt
import hashlib
from networkx.classes import graph
from trie_conceptnet import Trie
from trie_structure import Trie_hash
from Graph_builder import GraphBuilder
import spacy

def main():
    dictionary = json.load(open('/Users/rishabhsingh/Downloads/tumail.json', 'r'))
    trie_hash = Trie_hash()
    trie = Trie()
    gb = GraphBuilder()
    graph = nx.MultiGraph()
    node_creation(graph, trie_hash, dictionary,gb)
#    edge_creation(graph, gb, new_nodes)
    visualise_graph(graph)
#    save_graph(graph)
    save_graph_pickle(graph)

#    edge_creation(graph, dictionary,trie)


if __name__ == '__main__':
    main()