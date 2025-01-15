import pickle
from tqdm import tqdm
import re
import gensim.downloader
import torch_geometric.transforms as T

import copy
import torch
from geo_data_obj_directed import GeometricDataObject
from torch_geometric.nn import HGTConv, Linear


class LinkPrediction:
    def __init__(self, graph):
        self.graph = graph
        self.predicted_labels = None
        self.data_object = None
        self.make_data_object()
        self.process_data_before_training()
        self.model_prediction()


    def get_edges_of_type(self, src_type, edge_type, tgt_type):
        edges = []
        # Iterate through all edges and filter based on node types and edge types
        for u, v, data in self.graph.edges(data=True):
            # Check if the edge type and node types match
            if data.get('edge_type') == edge_type and \
                    self.graph.nodes[u].get('node_type') == src_type and \
                    self.graph.nodes[v].get('node_type') == tgt_type:
                edges.append({
                    'source': u,
                    'target': v,
                    'edge_type': data["edge_type"],
                    'is_predicted': data["is_predicted"]
                })

        return edges

    def assign_pred_values(self,predict_edges, predicted_labels):
        for i, edge in enumerate(predict_edges):
            edge["predicted_label"] = predicted_labels[i].item()
        return predict_edges

    def make_data_object(self):
        self.data_object = GeometricDataObject(self.graph).hetero_data

    def process_data_before_training(self):

        transform = T.RandomLinkSplit(
            num_val=0.0,
            num_test=0.0,
            disjoint_train_ratio=0.0,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=True,
            edge_types=("sentence", "mentions", "noun"),
            #rev_edge_types=("noun", "belongs_to", "email"),
        )

        train_data, val_data, test_data = transform(self.data_object)

        # do not need to split into edge_label_index for inference
        self.data_object.edge_label_index = train_data["sentence", "mentions", "noun"].edge_index

    def model_prediction(self):
        model = Model(hidden_channels=64, out_channels=12, num_heads=2, num_layers=2, metadata=self.data_object.metadata())
        model.load_state_dict(
            torch.load('/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/trained_model.pth'))
        model.eval()  # Switch to evaluation mode

        # Make inference on a new graph (or the same one, like `data_original`)
        with torch.no_grad():
            original_pred = model(self.data_object)

        # Example: Raw model output (logits)
        logits = original_pred  # The tensor you provided

        # Step 1: Apply sigmoid function to get probabilities
        probabilities = torch.sigmoid(logits)

        # Step 2: Apply a threshold to convert probabilities to binary predictions
        self.predicted_labels = (probabilities > 0.5).float()

        """predict_edges = self.get_edges_of_type(self.graph, "email", "belongs_to", "noun")
        predict_edges = self.assign_pred_values(predict_edges, predicted_labels)"""



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
        self.hgt = HGT(hidden_channels, out_channels, num_heads, num_layers, metadata)
        self.classifier = Classifier()

    def forward(self, data):
        # x_dict contains the feature matrices of all node types
        # edge_index_dict contains the edge indices for all edge types
        x_dict = self.hgt(data.x_dict, data.edge_index_dict)

        # Get predictions for "email belongs_to noun" edges using the classifier
        pred = self.classifier(
            x_dict["sentence"],
            x_dict["noun"],
            data.edge_label_index,
        )

        return pred

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        #email_features = x_dict.pop('email')
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
#        return self.lin(x_dict['context'])
        return x_dict