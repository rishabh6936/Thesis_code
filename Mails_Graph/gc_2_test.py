import networkx as nx
import json
import matplotlib.pyplot as plt
import hashlib
from networkx.classes import graph
from trie_conceptnet import Trie
from trie_structure import Trie_hash
from Graph_builder import GraphBuilder
from know_ex2 import KnowExtract
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os

path = '/Users/rishabhsingh/Shira_thesis/Crawlers/dia_trans_net/data/conceptNet_embs',
save_path = '/Users/rishabhsingh/PycharmProjects/Mails_Graph/datasets/',
emb_file_name = 'conceptnet_embs_eng'

def edge_creation(graph, gb,new_nodes):
    """First add a directed edge from Email to all its respective related nodes"""
    """for records in dictionary:
        email = records['Text']
#        unique_id = graph.nodes[email]['unique_id']
        edge_email_to_others(records,email,graph,trie)"""

#    email = new_nodes['Text']
    email = [value for key, value in new_nodes if key == 'Text']
    if isinstance(email, list):
        email = ''.join(email)  # Convert list to string

    email = 'Hello, wanna buy some weed?'

    msg_sub = KnowExtract(email,gb.trie, 2)
    for hop in range(4):
        ex_nodes = msg_sub.new_hop(hop_nr=hop, k=10)
        if ex_nodes == 0:
            break



def main():
#    dictionary = json.load(open('/Users/rishabhsingh/Downloads/tumail.json', 'r'))
#    trie_hash = Trie_hash()
#    trie = Trie()
    gb = GraphBuilder()
#    graph = nx.MultiGraph()
#    node_creation(graph, trie_hash, dictionary,gb)
    new_nodes = []
#    edge_creation(graph, dictionary, gb, new_nodes)
    edge_creation(graph, gb, new_nodes)
#    visualise_graph(graph)
#    edge_creation(graph, dictionary,trie)


if __name__ == '__main__':
    main()
