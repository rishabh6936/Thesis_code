import copy
import os.path as osp
import time
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils.convert import to_networkx, from_networkx

from torch_geometric.datasets import OGB_MAG

sample_dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
sample_data = sample_dataset[0]


hetero_data = HeteroData()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph.pkl'
with open(save_path, mode='rb') as f:
    dataset = pickle.load(f)

def get_nodetype_length(node_type):
    length = len([n for n, attr in dataset.nodes(data=True) if attr.get('node_type') == node_type])
    return length

def get_edgetype_length(src_type, tgt_type, edge_type):
    length = len([
        1 for source, target, attr in dataset.edges(data=True)
        if attr.get('edge_type') == edge_type
        and dataset.nodes[source]['node_type'] == src_type
        and dataset.nodes[target]['node_type'] == tgt_type
    ])
    return length

def process_graph(graph):
    node_dict = {node: i for i, node in enumerate(graph.nodes)}
    email_node_dict = get_node_type_indexes(graph,'email')
    adresse_node_dict = get_node_type_indexes(graph, 'adresse')
    betreff_node_dict = get_node_type_indexes(graph, 'Betreff')
    name_node_dict = get_node_type_indexes(graph, 'name')
    noun_node_dict = get_node_type_indexes(graph, 'noun')
    context_node_dict = get_node_type_indexes(graph, 'context')
    edge_index_belongs_to = get_edge_type_indexes(graph,'email','noun', node_dict)
    print(node_dict)

def get_node_type_indexes(graph,node_type):
    type_node_dict = {
        node: i
        for i, node in enumerate(graph.nodes)
        if node in dataset.nodes() and dataset.nodes[node].get('node_type') == node_type
    }

    return type_node_dict

def get_edge_type_indexes(graph,src_edge_type,tgt_edge_type,node_dict):
    edge_index = []
    source_indices = []
    target_indices = []
    for edge in graph.edges(data=True):
        source, target, attr = edge
        # Determine the edge type based on source and target node types
        source_type = graph.nodes[source].get('node_type')
        target_type = graph.nodes[target].get('node_type')

        if source_type == src_edge_type and target_type == tgt_edge_type:
            source_indices.append(node_dict[source])
            target_indices.append(node_dict[target])

    # Convert lists to tensors and stack them into the geometric format
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    return edge_index


process_graph(dataset)

pyg = from_networkx(dataset)
hetero_data= pyg

hetero_data['email'].x = [get_nodetype_length('email'), 384]
hetero_data['adresse'].x = [get_nodetype_length('adresse'), 384]
hetero_data['Betreff'].x = [get_nodetype_length('Betreff'), 384]
hetero_data['name'].x = [get_nodetype_length('name'), 384]
hetero_data['noun'].x = [get_nodetype_length('noun'), 384]
hetero_data['context'].x = [get_nodetype_length('context'), 384]

hetero_data['email', 'noun', 'belongs_to'].edge_index = [2, get_edgetype_length('email','noun','belongs_to')]
hetero_data['email', 'adresse', 'Von: (Adresse)'].edge_index = [2, get_edgetype_length('email','adresse','Von: (Adresse)')]
hetero_data['email', 'adresse', 'An: (Adresse)'].edge_index = [2, get_edgetype_length('email','adresse','An: (Adresse)')]
hetero_data['email', 'Betreff', 'Betreff'].edge_index = [2, get_edgetype_length('email','Betreff','Betreff')]
hetero_data['email', 'name', 'An: (Name)'].edge_index = [2, get_edgetype_length('email','name','An: (Name)')]
hetero_data['email', 'name', 'Von: (Name)'].edge_index = [2, get_edgetype_length('email','name','Von: (Name)')]
hetero_data['noun', 'context', 'has_context'].edge_index = [2, get_edgetype_length('noun','context','has_context')]
hetero_data['noun', 'sentence', 'comes_before'].edge_index = [2, get_edgetype_length('noun','sentence','comes_before')]
hetero_data['noun', 'sentence', 'comes_after'].edge_index = [2, get_edgetype_length('noun','sentence','comes_after')]

hetero_data['email', 'noun', 'belongs_to'].edge_attr = [get_edgetype_length('email','noun','belongs_to'),1]
hetero_data['email', 'adresse', 'Von: (Adresse)'].edge_attr = [get_edgetype_length('email','adresse','Von: (Adresse)'),1]
hetero_data['email', 'adresse', 'An: (Adresse)'].edge_attr = [get_edgetype_length('email','adresse','An: (Adresse)'),1]
hetero_data['email', 'Betreff', 'Betreff'].edge_attr = [get_edgetype_length('email','Betreff','Betreff'),1]
hetero_data['email', 'name', 'An: (Name)'].edge_attr = [get_edgetype_length('email','name','An: (Name)'),1]
hetero_data['email', 'name', 'Von: (Name)'].edge_attr = [get_edgetype_length('email','name','Von: (Name)'),1]
hetero_data['noun', 'context', 'has_context'].edge_attr = [get_edgetype_length('noun','context','has_context'),1]
hetero_data['noun', 'sentence', 'comes_before'].edge_attr = [get_edgetype_length('noun','sentence','comes_before'),1]
hetero_data['noun', 'sentence', 'comes_after'].edge_attr = [get_edgetype_length('noun','sentence','comes_after'),1]


node_types, edge_types = hetero_data.metadata()





dataset.train_mask = torch.zeros(len(dataset.nodes), dtype=torch.bool)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GNN(hidden_channels=64, out_channels=2)
model = to_hetero(model, hetero_data.metadata(), aggr='sum')

print(model)