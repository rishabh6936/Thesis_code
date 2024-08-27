import networkx as nx
import json
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
import os

path = '/Users/rishabhsingh/Shira_thesis/Crawlers/dia_trans_net/data/conceptNet_embs',
save_path = '/Users/rishabhsingh/PycharmProjects/Mails_Graph/datasets/',
emb_file_name = 'conceptnet_embs_eng'
def node_creation(graph, trie_hash, dictionary,gb):

    for records in dictionary:
        new_nodes = []
        for key, value in records.items():
            if value is not None:
                hash_value = hashing_function(value)
                x = trie_hash.query(hash_value)        #if the node not present in the Graph
                if x == []:
                    graph.add_node(value, node_type=key, unique_id=hash_value)     #add node to the graph
                    set_trie_hash(trie_hash, hash_value, value)                              #add node to the Trie
                    new_nodes.append([key,value])                                  #make a list of nodes to make edges with
                if x != []:
                    new_nodes.append([key,value])
        edge_creation(graph, dictionary,gb,new_nodes)                            #send the new nodes for edge creation


def edge_creation(graph, dictionary,gb,new_nodes):
    """First add a directed edge from Email to all its respective related nodes"""
    """for records in dictionary:
        email = records['Text']
#        unique_id = graph.nodes[email]['unique_id']
        edge_email_to_others(records,email,graph,trie)"""

#    email = new_nodes['Text']
    email = [value for key, value in new_nodes if key == 'Text']
    if isinstance(email, list):
        email = ''.join(email)  # Convert list to string

    msg_sub = KnowExtract(email,gb.trie, 2)
    for hop in range(4):
        ex_nodes = msg_sub.new_hop(hop_nr=hop, k=10)
        if ex_nodes == 0:
            break

    for key, value in new_nodes:                           #add normal nodes from an email for example Betreff
        if key != "Text":
            graph.add_edge(email, value, edge_type=key)

    for entity in msg_sub.graph_edges:
        graph.add_edge(email, entity[0], edge_type='belongs_to')
        graph.add_edge(entity[0], entity[1], edge_type=entity[2])

    for entity in msg_sub.graph_edges:
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
                graph.add_edge(email, node_after, edge_type='belongs_to')









"""def edge_email_to_others(records,unique_id,email,graph,trie):
    for key, value in records.items():
        if value is not None:
            x = Trie.query(trie,graph.nodes[value]['unique_id'])
            if graph.nodes[value]['unique_id'] != unique_id and value != email:
               graph.add_edge(email, value)"""

def set_trie_hash(trie_hash, hash_value, value):
    print("build trie hash datastructure")
    trie_hash.insert_dataset(hash_value, value)


def hashing_function(value):                    #hashing to asign a unique Id to each node
    hash_object = hashlib.sha256()
    hash_object.update(value.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    return hash_digest


def visualise_graph(graph):
    nx.draw(graph, with_labels=False)
    plt.show()



def main():
    dictionary = json.load(open('/Users/rishabhsingh/Downloads/tumail.json', 'r'))
    trie_hash = Trie_hash()
    trie = Trie()
    gb = GraphBuilder()
    graph = nx.MultiGraph()
    node_creation(graph, trie_hash, dictionary,gb)
    visualise_graph(graph)
#    edge_creation(graph, dictionary,trie)


if __name__ == '__main__':
    main()
