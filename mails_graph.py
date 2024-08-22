import networkx as nx
import json
import matplotlib.pyplot as plt
import hashlib
from networkx.classes import graph
from trie_structure import Trie


def node_creation(graph, trie, dictionary):

    for records in dictionary:
        new_nodes = []
        for key, value in records.items():
            if value is not None:
                hash_value = hashing_function(value)
                x = Trie.query(trie, hash_value)        #if the node not present in the Graph
                if x == []:
                    graph.add_node(value, node_type=key, unique_id=hash_value)     #add node to the graph
                    set_trie(trie, hash_value, value)                              #add node to the Trie
                    new_nodes.append([key,value])                                  #make a list of nodes to make edges with
                if x != []:
                    new_nodes.append([key,value])
        edge_creation(graph, dictionary,trie,new_nodes)                            #send the new nodes for edge creation


def edge_creation(graph, dictionary,trie,new_nodes):
    """First add a directed edge from Email to all its respective reslated nodes"""
    """for records in dictionary:
        email = records['Text']
#        unique_id = graph.nodes[email]['unique_id']
        edge_email_to_others(records,email,graph,trie)"""

#    email = new_nodes['Text']
    email = [value for key, value in new_nodes if key == 'Text']
    if isinstance(email, list):
        email = ''.join(email)  # Convert list to string
    for key, value in new_nodes:
        if key != "Text":
            graph.add_edge(email, value, edge_type=key)




"""def edge_email_to_others(records,unique_id,email,graph,trie):
    for key, value in records.items():
        if value is not None:
            x = Trie.query(trie,graph.nodes[value]['unique_id'])
            if graph.nodes[value]['unique_id'] != unique_id and value != email:
               graph.add_edge(email, value)"""

def set_trie(trie, hash_value, value):
    print("build trie datastructure")
    Trie.insert_dataset(trie, hash_value, value)


def hashing_function(value):                    #hashing to asign a unique Id to each node
    hash_object = hashlib.sha256()
    hash_object.update(value.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    return hash_digest


def visualise_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()


def main():
    dictionary = json.load(open('/Users/rishabhsingh/Downloads/tumail.json', 'r'))
    trie = Trie()
    graph = nx.MultiGraph()
    node_creation(graph, trie, dictionary)
#    visualise_graph(graph)
#    edge_creation(graph, dictionary,trie)


if __name__ == '__main__':
    main()
