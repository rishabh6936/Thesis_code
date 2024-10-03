import torch
import pickle
import networkx as nx
from torch_geometric.data import HeteroData
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph.pkl'
with open(save_path, mode='rb') as f:
    graph = pickle.load(f)"""

class GeometricDataObject:
        def __init__(self,graph):
            self.graph = graph
            self.node_types = set(nx.get_node_attributes(graph, 'node_type').values())
            self.hetero_data = HeteroData()
            self.process_graph()
            self.make_hetero_object()

        def get_nodetype_length(self,node_type):
            length = len([n for n, attr in self.graph.nodes(data=True) if attr.get('node_type') == node_type])
            return length


        def get_edgetype_length(self,src_type, tgt_type, edge_type):
            length = len([
                1 for source, target, attr in self.graph.edges(data=True)
                if attr.get('edge_type') == edge_type
                   and self.graph.nodes[source]['node_type'] == src_type
                   and self.graph.nodes[target]['node_type'] == tgt_type
            ])
            return length


        def process_graph(self):
            node_dict = {node: i for i, node in enumerate(self.graph.nodes)}
            self.graph.email_node_dict = self.get_node_type_indexes( 'email')
            self.graph.adresse_node_dict = self.get_node_type_indexes( 'adresse')
            self.graph.betreff_node_dict = self.get_node_type_indexes( 'betreff')
            self.graph.name_node_dict = self.get_node_type_indexes( 'name')
            self.graph.noun_node_dict = self.get_node_type_indexes('noun')
            self.graph.context_node_dict = self.get_node_type_indexes( 'context')
            self.graph.sentence_node_dict = self.get_node_type_indexes('sentence')

            self.graph.ei_belongs_to = self.get_edge_type_indexes('email', 'noun', 'belongs_to', node_dict)
            self.graph.ei_belongs_to = self.map_to_local_indices(self.graph.ei_belongs_to)

            self.graph.ei_an_adresse = self.get_edge_type_indexes( 'email', 'adresse', 'sent_to', node_dict)
            self.graph.ei_an_adresse = self.map_to_local_indices(self.graph.ei_an_adresse)

            self.graph.ei_von_adresse = self.get_edge_type_indexes( 'email', 'adresse', 'sent_from', node_dict)
            self.graph.ei_von_adresse = self.map_to_local_indices(self.graph.ei_von_adresse)

            self.graph.ei_betreff = self.get_edge_type_indexes( 'email', 'betreff', 'title', node_dict)
            self.graph.ei_betreff = self.map_to_local_indices(self.graph.ei_betreff)

            self.graph.ei_an_name = self.get_edge_type_indexes('email', 'name', 'to_person', node_dict)
            self.graph.ei_an_name = self.map_to_local_indices(self.graph.ei_an_name)

            self.graph.ei_von_name = self.get_edge_type_indexes('email', 'name', 'from_person', node_dict)
            self.graph.ei_von_name = self.map_to_local_indices(self.graph.ei_von_name)

            self.graph.ei_context = self.get_edge_type_indexes('noun', 'context', 'has_context', node_dict)
            self.graph.ei_context = self.map_to_local_indices(self.graph.ei_context)

            self.graph.ei_comes_before = self.get_edge_type_indexes('noun', 'sentence', "['comes_before']", node_dict)
            self.graph.ei_comes_before = self.map_to_local_indices(self.graph.ei_comes_before)

            self.graph.ei_comes_after = self.get_edge_type_indexes('noun', 'sentence', "['comes_after']", node_dict)
            self.graph.ei_comes_after = self.map_to_local_indices(self.graph.ei_comes_after)

        def make_hetero_object(self):
            self.hetero_data['email'].x = self.get_node_feature_matrix(self.graph.email_node_dict)
            self.hetero_data['adresse'].x = self.get_node_feature_matrix(self.graph.adresse_node_dict)
            self.hetero_data['betreff'].x = self.get_node_feature_matrix(self.graph.betreff_node_dict)
            self.hetero_data['name'].x = self.get_node_feature_matrix(self.graph.name_node_dict)
            self.hetero_data['context'].x = self.get_node_feature_matrix(self.graph.context_node_dict)
            self.hetero_data['sentence'].x = self.get_node_feature_matrix(self.graph.sentence_node_dict)
            self.hetero_data['noun'].x = self.get_node_feature_matrix(self.graph.noun_node_dict)

            # Set the edge indices for email to noun relationships
            self.hetero_data['email', 'belongs_to', 'noun'].edge_index = self.graph.ei_belongs_to
            self.hetero_data['noun', 'belongs_to', 'email'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_belongs_to)

            # Set the edge indices for email to adresse relationships
            self.hetero_data['email', 'sent_to', 'adresse'].edge_index = self.graph.ei_an_adresse
            self.hetero_data['adresse', 'sent_to', 'email'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_an_adresse)

            self.hetero_data['email', 'sent_from', 'adresse'].edge_index = self.graph.ei_von_adresse
            self.hetero_data['adresse', 'sent_from', 'email'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_von_adresse)

            # Set the edge indices for email to betreff relationships
            self.hetero_data['email', 'title', 'betreff'].edge_index = self.graph.ei_betreff
            self.hetero_data['betreff', 'title', 'email'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_betreff)

            # Set the edge indices for email to name relationships
            self.hetero_data['email', 'to_person', 'name'].edge_index = self.graph.ei_an_name
            self.hetero_data['name', 'to_person', 'email'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_an_name)

            # Uncomment if needed
            # self.hetero_data['email', 'from_person', 'name'].edge_index = self.graph.ei_von_name
            # self.hetero_data['name', 'from_person', 'email'].edge_index = self.reverse_map_to_local_indices(self.graph.ei_von_name)

            # Set the edge indices for noun to context relationships
            self.hetero_data['noun', 'has_context', 'context'].edge_index = self.graph.ei_context
            self.hetero_data['context', 'has_context', 'noun'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_context)

            # Set the edge indices for noun to sentence relationships
            self.hetero_data['noun', 'comes_before', 'sentence'].edge_index = self.graph.ei_comes_before
            self.hetero_data['sentence', 'comes_before', 'noun'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_comes_before)

            self.hetero_data['noun', 'comes_after', 'sentence'].edge_index = self.graph.ei_comes_after
            self.hetero_data['sentence', 'comes_after', 'noun'].edge_index = self.reverse_map_to_local_indices(
                self.graph.ei_comes_after)

            """hetero_data.edge_types = [('email','noun', 'belongs_to'), ('email','adresse','Von: (Adresse)'),
                                      ('email','adresse','An: (Adresse)'), ('email','Betreff','Betreff'),
                                      ('email','name','An: (Name)'), ('email','name','Von: (Name)'),
                                      ('email','name','Von: (Name)'), ('noun','context','has_context'),
                                      ('noun','sentence',"['comes_before']"), ('noun','sentence',"['comes_after']")
                                      ]"""

        #def get_local_node_dict():


        def map_to_local_indices(self, edge_index):
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

        def reverse_map_to_local_indices(self, edge_index):
            local_source_indices = edge_index[0].flatten().tolist()
            local_target_indices = edge_index[1].flatten().tolist()
            new_edge_index = torch.tensor([local_target_indices, local_source_indices], dtype=torch.long)
            return new_edge_index

        def get_node_type_indexes(self, node_type):
            type_node_dict = {
                node: i for i, node in enumerate(self.graph.nodes)
                if node in self.graph.nodes() and self.graph.nodes[node].get('node_type') == node_type
            }

            return type_node_dict


        def get_node_feature_matrix(self,type_node_dict):
            embeddings_matrix = []
            for node, idx in type_node_dict.items():
                embedding = self.graph.nodes[node].get('embedding')
                embeddings_matrix.append(embedding)

            embedding_tensor = torch.from_numpy(np.asarray(embeddings_matrix))
            return embedding_tensor


        def get_edge_type_indexes(self, src_edge_type, tgt_edge_type, edge_type, node_dict):
            edge_index = []
            source_indices = []
            target_indices = []
            for edge in self.graph.edges(data=True):
                source, target, attr = edge
                # Determine the edge type based on source and target node types
                source_type = self.graph.nodes[source].get('node_type')
                target_type = self.graph.nodes[target].get('node_type')
                attr_type = attr['edge_type']

                if source_type == src_edge_type and target_type == tgt_edge_type and edge_type == attr_type:
                    source_indices.append(node_dict[source])
                    target_indices.append(node_dict[target])

            # Convert lists to tensors and stack them into the geometric format
            edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
            return edge_index



