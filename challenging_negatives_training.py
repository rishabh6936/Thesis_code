import pickle
from tqdm import tqdm
import re
import gensim.downloader
import torch_geometric.transforms as T
from nltk import RegexpTokenizer
from geometric_data_object import GeometricDataObject
import copy
import torch
from sentence_transformers import SentenceTransformer
import gc
import fasttext
import fasttext.util
from torch_geometric.nn import HGTConv, Linear

# Download and load the fasttext model
fasttext.util.download_model('de', if_exists='ignore')  # German model
ft = fasttext.load_model('cc.de.300.bin')

# gb = GraphBuilder()
# Load pre-trained word vectors (e.g., 'glove-wiki-gigaword-100' or 'word2vec-google-news-300')
# glove_vectors = gensim.downloader.load('word2vec-google-news-300')  # or 'word2vec-google-news-300'
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.to(torch.device('cpu'))

save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph_small.pkl'
with open(save_path, mode='rb') as f:
    graph = pickle.load(f)


def get_node_embedding(node):
    embeddings = model.encode(node)
    return embeddings


def get_corrupted_embedding(word, corruption_type='misspelled'):
    if corruption_type == 'misspelled':
        corrupted_word = keyboard_typo(word)
    else:  # semantic corruption
        corrupted_word = get_closest_word(word)
    corrupted_embedding = get_node_embedding(corrupted_word)
    return corrupted_embedding, corrupted_word


def keyboard_typo(word):
    import random
    keyboard = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfc', 'e': 'rdsw',
        'f': 'rtgv', 'g': 'tyhb', 'h': 'yujn', 'i': 'ujko', 'k': 'ijml',
        'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'plki', 'p': 'ol',
        'q': 'wa', 'r': 'tfd', 's': 'wedx', 't': 'ygfr', 'u': 'yihj',
        'v': 'cfgb', 'w': 'qase', 'x': 'sdcz', 'y': 'uhgt', 'z': 'asx'
    }
    idx = random.randint(0, len(word) - 1)
    char = word[idx].lower()
    if char in keyboard:
        typo_char = random.choice(keyboard[char])
        word = word[:idx] + typo_char + word[idx + 1:]
    return word


def get_closest_word(word):
    # Check if the word exists in the GloVe vectors' vocabulary

    similar_words = ft.get_nearest_neighbors(word)
    closest_word = similar_words[1][1] if similar_words else word
    return closest_word

def replace_word(old_word, new_word, email):
    tokenizer = RegexpTokenizer(r'\d+,\d+[a-zA-Z]|\w+')
    split_sentence = tokenizer.tokenize(email)

#    split_sentence = sentence.split()
    replaced_sentence = []
    for word in split_sentence:
        if word == old_word:
            replaced_sentence.append(new_word)
        else:
            replaced_sentence.append(word)

    replaced_sentence = ' '.join(replaced_sentence)
    return replaced_sentence


def get_sentence_in_email(word, email):
    sentence_pattern = r'([^.?!]*[.?!])'
    sentences = re.findall(sentence_pattern, email)
    word_similarity_data = []
    sentence_hit = ''
    # Iterate through sentences and check if the entity is present
    #    word = word[0].split('/')[3]

    entity_pattern = re.compile(rf'\b{re.escape(word)}\b')

    for sentence in sentences:
        if re.search(entity_pattern, sentence):
            sentence_hit = sentence.strip()
            break
    return sentence_hit


# geometric_data_object = GeometricDataObject(graph)
email_nodes = [node for node, attr in graph.nodes(data=True) if attr.get('node_type') == 'email']

"""suitable_edges = []
for email in email_nodes:
    email_edges = graph.edges(email, data=True)
    for edge in email_edges:
        source, target, attr = edge
        # Determine the edge type based on source and target node types
        source_type = graph.nodes[source].get('node_type')
        target_type = graph.nodes[target].get('node_type')
        attr_type = attr['edge_type']

        if source_type == 'email' and target_type == 'noun' and attr_type == 'belongs_to':
           suitable_edges.append(edge)"""

"""for edges in suitable_edges:
    noun = edges[1]
    msg_sub = KnowExtract(edges[0], gb.trie, 2)
    for nodes in msg_sub.data['knowledge_nodes']:
        corrupt_word = keyboard_typo(nodes)
        sentence_hit = get_sentence_in_email(nodes, edges[0])
        #using this function to actually replace the misspelled word in place of correct
        if sentence_hit != '':
           replaced_sentence = replace_misspelled_candidate(nodes, corrupt_word, sentence_hit)
        most_similar_word = glove_vectors.most_similar(nodes)"""

"""def find_similar_word(word):
    try:
        most_similar = glove_vectors.most_similar(word.lower(), topn=1)
        return most_similar[0][0]
    except KeyError:
        # If the word isn't in the vocabulary, return the original word
        return word"""


def create_corrupted_graphs(graph, email_nodes):
    # Create deep copies of the original graph
    typo_graph = copy.deepcopy(graph)
    similar_word_graph = copy.deepcopy(graph)

    for email in tqdm(email_nodes, total=len(email_nodes), desc='Processing emails'):
        email_edges = graph.edges(email, data=True)
        for edge in email_edges:
            source, target, attr = edge
            if graph.nodes[source]['node_type'] == 'email' and graph.nodes[target]['node_type'] == 'noun' and attr[
                'edge_type'] == 'belongs_to':
                noun_word = target  # Assuming 'target' is the noun

                # Replace with misspelled version
                misspelled_embedding, misspelled_word = get_corrupted_embedding(noun_word, 'misspelled')
                new_email_node = replace_word(noun_word, misspelled_word, email)
                typo_graph.add_node(new_email_node, node_type='email', embedding=get_node_embedding(new_email_node))
                typo_graph.add_node(misspelled_word, node_type='noun', embedding=misspelled_embedding)
                typo_graph.add_edge(new_email_node, misspelled_word, edge_type='belongs_to')
                #                typo_graph.add_node(misspelled_word, node_type='noun', embedding=misspelled_embedding)
                #typo_graph.nodes[noun_word]['embedding'] = misspelled_embedding
                #typo_graph.nodes[noun_word]['corrupted_as'] = misspelled_word
                # Add edges with corrupt nouns
                #                typo_graph.add_edge(source, misspelled_word, edge_type='belongs_to')

                # Replace with semantic corruption
                similar_embedding, similar_word = get_corrupted_embedding(noun_word, 'semantic')
                if similar_word != '':
                    #                   similar_word_graph.add_node(similar_word, node_type='noun', embedding=similar_embedding)
                    new_email_node = replace_word(noun_word, misspelled_word,email)
                    similar_word_graph.add_node(new_email_node, node_type='email', embedding=get_node_embedding(new_email_node))
                    similar_word_graph.add_node(similar_word, node_type='noun', embedding=similar_embedding)
                    similar_word_graph.add_edge(new_email_node, similar_word, edge_type='belongs_to')
                    #similar_word_graph.nodes[noun_word]['embedding'] = similar_embedding
                    #similar_word_graph.nodes[noun_word]['corrupted_as'] = similar_word
    #                   similar_word_graph.add_edge(source, similar_word, edge_type='belongs_to')

    #    gc.collect()
    return typo_graph, similar_word_graph


def preprocess_graphs(typo_graph, similar_word_graph, graph):
    data_typograph = GeometricDataObject(typo_graph)
    data_typograph = data_typograph.hetero_data
    data_similar_word_graph = GeometricDataObject(similar_word_graph)
    data_similar_word_graph = data_similar_word_graph.hetero_data
    data_original = GeometricDataObject(graph)
    data_original = data_original.hetero_data
    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        disjoint_train_ratio=0,
        neg_sampling_ratio=0,
        add_negative_train_samples=False,
        edge_types=("email", "belongs_to", "noun"),
        rev_edge_types=("noun", "belongs_to", "email"),
    )
    train_data_typo, val_data_typo, test_data_typo = transform(data_typograph)
    train_data_sim, val_data_typo_sim, test_data_typo_sim = transform(data_similar_word_graph)
    train_data, val_data, test_data = transform(data_original)

    data_original.edge_label_index = train_data["email", "belongs_to", "noun"].edge_label_index
    data_original["email", "belongs_to", "noun"].edge_label = train_data[
        "email", "belongs_to", "noun"].edge_label  # positive Example

    data_typograph.edge_label_index = train_data_typo["email", "belongs_to", "noun"].edge_label_index

    # groundtruth label 0, negative example
    data_typograph["email", "belongs_to", "noun"].edge_label = torch.zeros_like(
        train_data_typo["email", "belongs_to", "noun"].edge_label)

    data_similar_word_graph.edge_label_index = train_data_sim["email", "belongs_to", "noun"].edge_label_index

    # groundtruth label 0, negative example
    data_similar_word_graph["email", "belongs_to", "noun"].edge_label = torch.zeros_like(
        train_data_sim["email", "belongs_to", "noun"].edge_label)

    # Instantiate the model
    model = Model(hidden_channels=64, out_channels=12, num_heads=2, num_layers=2, metadata=data_typograph.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_with_corruptions(model, data_typograph, data_similar_word_graph, data_original, optimizer, criterion)
    # Save the model after training
    torch.save(model.state_dict(), '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/trained_model.pth')


def train_with_corruptions(model, data_typograph, data_similar_word_graph, data_original, optimizer, criterion,
                           epochs=100):
    for epoch in range(epochs):
        # Step 1: Train on the typo graph
        optimizer.zero_grad()
        typo_pred = model(data_typograph)
        typo_ground_truth = data_typograph["email", "belongs_to", "noun"].edge_label  # Example label extraction
        typo_loss = criterion(typo_pred, typo_ground_truth)
        typo_loss.backward()
        optimizer.step()

        # Step 2: Train on the similar word graph
        optimizer.zero_grad()
        similar_pred = model(data_similar_word_graph)
        similar_ground_truth = data_similar_word_graph["email", "belongs_to", "noun"].edge_label
        similar_loss = criterion(similar_pred, similar_ground_truth)
        similar_loss.backward()
        optimizer.step()

        # Step 3: Train on the original graph
        optimizer.zero_grad()
        original_pred = model(data_original)
        original_ground_truth = data_original["email", "belongs_to", "noun"].edge_label
        original_loss = criterion(original_pred, original_ground_truth)
        original_loss.backward()
        optimizer.step()

        # Print the loss for each corrupted graph
        print(f"Epoch {epoch + 1}/{epochs}, Typo Loss: {typo_loss.item()}, Similar Word Loss: {similar_loss.item()},Original Word Loss: {original_loss.item()} ")


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
            x_dict["email"],
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

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
#        return self.lin(x_dict['context'])
        return x_dict