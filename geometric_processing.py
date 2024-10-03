import copy
import os.path as osp
import time
import pickle
from torch_geometric.nn import HGTConv, Linear
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import HeteroData
import networkx as nx
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils.convert import to_networkx, from_networkx
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG

sample_dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
sample_data = sample_dataset[0]

hetero_data = HeteroData()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph.pkl'
with open(save_path, mode='rb') as f:
    graph = pickle.load(f)

node_types = set(nx.get_node_attributes(graph, 'node_type').values())
edge_types = set(nx.get_edge_attributes(graph, 'edge_type').values())

# Split edge set for training and testing
"""edges = graph.edges()

eids = np.arange(graph.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = graph.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]"""

def get_nodetype_length(node_type):
    length = len([n for n, attr in graph.nodes(data=True) if attr.get('node_type') == node_type])
    return length


def get_edgetype_length(src_type, tgt_type, edge_type):
    length = len([
        1 for source, target, attr in graph.edges(data=True)
        if attr.get('edge_type') == edge_type
           and graph.nodes[source]['node_type'] == src_type
           and graph.nodes[target]['node_type'] == tgt_type
    ])
    return length


def process_graph(graph):
    node_dict = {node: i for i, node in enumerate(graph.nodes)}
    graph.email_node_dict = get_node_type_indexes(graph, 'email')
    graph.adresse_node_dict = get_node_type_indexes(graph, 'adresse')
    graph.betreff_node_dict = get_node_type_indexes(graph, 'betreff')
    graph.name_node_dict = get_node_type_indexes(graph, 'name')
    graph.noun_node_dict = get_node_type_indexes(graph, 'noun')
    graph.context_node_dict = get_node_type_indexes(graph, 'context')
    graph.sentence_node_dict = get_node_type_indexes(graph, 'sentence')

    graph.ei_belongs_to = get_edge_type_indexes(graph, 'email', 'noun', 'belongs_to', node_dict)
    graph.ei_belongs_to = map_to_local_indices(graph.ei_belongs_to)

    graph.ei_an_adresse = get_edge_type_indexes(graph, 'email', 'adresse', 'sent_to', node_dict)
    graph.ei_an_adresse = map_to_local_indices(graph.ei_an_adresse)

    graph.ei_von_adresse = get_edge_type_indexes(graph, 'email', 'adresse', 'sent_from', node_dict)
    graph.ei_von_adresse = map_to_local_indices(graph.ei_von_adresse)

    graph.ei_betreff = get_edge_type_indexes(graph, 'email', 'betreff', 'title', node_dict)
    graph.ei_betreff = map_to_local_indices(graph.ei_betreff)

    graph.ei_an_name = get_edge_type_indexes(graph, 'email', 'name', 'to_person', node_dict)
    graph.ei_an_name = map_to_local_indices(graph.ei_an_name)

    graph.ei_von_name = get_edge_type_indexes(graph, 'email', 'name', 'from_person', node_dict)
    graph.ei_von_name = map_to_local_indices(graph.ei_von_name)

    graph.ei_context = get_edge_type_indexes(graph, 'noun', 'context', 'has_context', node_dict)
    graph.ei_context = map_to_local_indices(graph.ei_context)

    graph.ei_comes_before = get_edge_type_indexes(graph, 'noun', 'sentence', "['comes_before']", node_dict)
    graph.ei_comes_before = map_to_local_indices(graph.ei_comes_before)

    graph.ei_comes_after = get_edge_type_indexes(graph, 'noun', 'sentence', "['comes_after']", node_dict)
    graph.ei_comes_after = map_to_local_indices(graph.ei_comes_after)



def make_hetero_object(graph):
    hetero_data['email'].x = get_node_feature_matrix(graph.email_node_dict)
    hetero_data['adresse'].x = get_node_feature_matrix(graph.adresse_node_dict)
    hetero_data['betreff'].x = get_node_feature_matrix(graph.betreff_node_dict)
    hetero_data['name'].x = get_node_feature_matrix(graph.name_node_dict)
    hetero_data['context'].x = get_node_feature_matrix(graph.context_node_dict)
    hetero_data['sentence'].x = get_node_feature_matrix(graph.sentence_node_dict)
    hetero_data['noun'].x = get_node_feature_matrix(graph.noun_node_dict)

    # edge types are identified by using a triplet (source_node_type, edge_type, destination_node_type)
    # Set the edge indices for the email to noun relationships
    hetero_data['email', 'belongs_to', 'noun'].edge_index = graph.ei_belongs_to
    hetero_data['noun', 'belongs_to', 'email'].edge_index = reverse_map_to_local_indices(graph.ei_belongs_to)

    # Set the edge indices for the email to adresse relationships
    hetero_data['email', 'sent_to', 'adresse'].edge_index = graph.ei_an_adresse
    hetero_data['adresse', 'sent_to', 'email'].edge_index = reverse_map_to_local_indices(graph.ei_an_adresse)

    hetero_data['email', 'sent_from', 'adresse'].edge_index = graph.ei_von_adresse
    hetero_data['adresse', 'sent_from', 'email'].edge_index = reverse_map_to_local_indices(graph.ei_von_adresse)

    # Set the edge indices for the email to betreff relationships
    hetero_data['email', 'title', 'betreff'].edge_index = graph.ei_betreff
    hetero_data['betreff', 'title', 'email'].edge_index = reverse_map_to_local_indices(graph.ei_betreff)

    # Set the edge indices for the email to name relationships
    hetero_data['email', 'to_person', 'name'].edge_index = graph.ei_an_name
    hetero_data['name', 'to_person', 'email'].edge_index = reverse_map_to_local_indices(graph.ei_an_name)

#    hetero_data['email', 'from_person', 'name'].edge_index = graph.ei_von_name
#    hetero_data['name', 'from_person', 'email'].edge_index = reverse_map_to_local_indices(graph.ei_von_name)

    # Set the edge indices for the noun to context relationships
    hetero_data['noun', 'has_context', 'context'].edge_index = graph.ei_context
    hetero_data['context', 'has_context', 'noun'].edge_index = reverse_map_to_local_indices(graph.ei_context)

    # Set the edge indices for the noun to sentence relationships
    hetero_data['noun', 'comes_before', 'sentence'].edge_index = graph.ei_comes_before
    hetero_data['sentence', 'comes_before', 'noun'].edge_index = reverse_map_to_local_indices(graph.ei_comes_before)

    hetero_data['noun', 'comes_after', 'sentence'].edge_index = graph.ei_comes_after
    hetero_data['sentence', 'comes_after', 'noun'].edge_index = reverse_map_to_local_indices(graph.ei_comes_after)

    """hetero_data.edge_types = [('email','noun', 'belongs_to'), ('email','adresse','Von: (Adresse)'),
                              ('email','adresse','An: (Adresse)'), ('email','Betreff','Betreff'),
                              ('email','name','An: (Name)'), ('email','name','Von: (Name)'),
                              ('email','name','Von: (Name)'), ('noun','context','has_context'),
                              ('noun','sentence',"['comes_before']"), ('noun','sentence',"['comes_after']")
                              ]"""

#def get_local_node_dict():


def map_to_local_indices(edge_index):
    # Flatten the edge index tensor to a 1D list of values
    all_indices_src = edge_index[0].flatten().tolist()
    all_indices_tgt = edge_index[1].flatten().tolist()

    # Create a dictionary to map global indices to local indices
    unique_indices_src = list(set(all_indices_src))  # Get unique values
    index_mapping_src = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted(unique_indices_src))}

    unique_indices_tgt = list(set(all_indices_tgt))  # Get unique values
    index_mapping_tgt = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted(unique_indices_tgt))}

    # Map the global indices to local indices using the mapping
    local_source_indices = [index_mapping_src[global_idx] for global_idx in edge_index[0].tolist()]
    local_target_indices = [index_mapping_tgt[global_idx] for global_idx in edge_index[1].tolist()]

    # Convert lists back to a tensor in the same shape as original edge_index
    new_edge_index = torch.tensor([local_source_indices, local_target_indices], dtype=torch.long)

    return new_edge_index  # Return new edge_index and the mapping

def reverse_map_to_local_indices(edge_index):
    local_source_indices = edge_index[0].flatten().tolist()
    local_target_indices = edge_index[1].flatten().tolist()
    new_edge_index = torch.tensor([local_target_indices, local_source_indices], dtype=torch.long)
    return new_edge_index

def get_node_type_indexes(graph, node_type):
    type_node_dict = {
        node: i for i, node in enumerate(graph.nodes)
        if node in graph.nodes() and graph.nodes[node].get('node_type') == node_type
    }

    return type_node_dict


def get_node_feature_matrix(type_node_dict):
    embeddings_matrix = []
    for node, idx in type_node_dict.items():
        embedding = graph.nodes[node].get('embedding')
        embeddings_matrix.append(embedding)

    embedding_tensor = torch.from_numpy(np.asarray(embeddings_matrix))
    return embedding_tensor


def get_edge_type_indexes(graph, src_edge_type, tgt_edge_type, edge_type, node_dict):
    edge_index = []
    source_indices = []
    target_indices = []
    for edge in graph.edges(data=True):
        source, target, attr = edge
        # Determine the edge type based on source and target node types
        source_type = graph.nodes[source].get('node_type')
        target_type = graph.nodes[target].get('node_type')
        attr_type = attr['edge_type']

        if source_type == src_edge_type and target_type == tgt_edge_type and edge_type == attr_type:
            source_indices.append(node_dict[source])
            target_indices.append(node_dict[target])

    # Convert lists to tensors and stack them into the geometric format
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    return edge_index


process_graph(graph)
make_hetero_object(graph)
pyg = from_networkx(graph)

"""hetero_data['email'].x = [get_nodetype_length('email'), 384]
hetero_data['adresse'].x = [get_nodetype_length('adresse'), 384]
hetero_data['Betreff'].x = [get_nodetype_length('Betreff'), 384]
hetero_data['name'].x = [get_nodetype_length('name'), 384]
hetero_data['noun'].x = [get_nodetype_length('noun'), 384]
hetero_data['context'].x = [get_nodetype_length('context'), 384]"""

"""hetero_data['email', 'noun', 'belongs_to'].edge_index = [2, get_edgetype_length('email','noun','belongs_to')]
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
hetero_data['noun', 'sentence', 'comes_after'].edge_attr = [get_edgetype_length('noun','sentence','comes_after'),1]"""

# node_types, edge_types = hetero_data.metadata()


graph.train_mask = torch.zeros(len(graph.nodes), dtype=torch.bool)

sample_dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
sample_data = sample_dataset[0]

transform = T.ToUndirected()
hetero_data_transformed = transform(hetero_data)


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in hetero_data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, hetero_data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
#        return self.lin(x_dict['context'])
        return x_dict

"""model = HGT(hidden_channels=64, out_channels=12,
            num_heads=2, num_layers=2)"""

hetero_data.validate()
"""with torch.no_grad():  # Initialize lazy modules.
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict)"""

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("email", "belongs_to", "noun"),
    rev_edge_types=("noun", "belongs_to", "email"),
)
train_data, val_data, test_data = transform(hetero_data)

# Define seed edges:
edge_label_index = train_data["email", "belongs_to", "noun"].edge_label_index
hetero_data["email", "belongs_to", "noun"].edge_label = train_data["email", "belongs_to", "noun"].edge_label
"""train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("email", "belongs_to", "noun"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)"""

class Classifier(torch.nn.Module):
    def forward(self, x_email: torch.Tensor, x_noun: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_email = x_email[edge_label_index[0]]
        edge_feat_noun = x_noun[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_email * edge_feat_noun).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        # HGT for heterogeneous graphs
        self.hgt = HGT(hidden_channels, out_channels, num_heads, num_layers)
        self.classifier = Classifier()

    def forward(self, data):
        # x_dict contains the feature matrices of all node types
        # edge_index_dict contains the edge indices for all edge types
        x_dict = self.hgt(data.x_dict, data.edge_index_dict)

        # Get predictions for "email belongs_to noun" edges using the classifier
        pred = self.classifier(
            x_dict["email"],
            x_dict["noun"],
            edge_label_index,
        )

        return pred

# Instantiate the model
model = Model(hidden_channels=64, out_channels=12, num_heads=2, num_layers=2, metadata=hetero_data.metadata())
# Initialize lazy modules
"""with torch.no_grad():
    out = model(hetero_data)"""

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 100):
        total_loss = total_examples = 0
        optimizer.zero_grad()
        pred = model(hetero_data)
        ground_truth = hetero_data["email", "belongs_to", "noun"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

#print(hetero_data)