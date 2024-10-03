import pickle
from graph_creation import replace_misspelled_candidate
import re
import gensim.downloader
from Graph_builder import GraphBuilder
from knowledge_extractor import KnowExtract
from geometric_data_object import GeometricDataObject
import copy
import torch
from sentence_transformers import SentenceTransformer

#gb = GraphBuilder()
# Load pre-trained word vectors (e.g., 'glove-wiki-gigaword-100' or 'word2vec-google-news-300')
glove_vectors = gensim.downloader.load('word2vec-google-news-300')  # or 'word2vec-google-news-300'
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.to(torch.device('cpu'))

save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph.pkl'
with open(save_path, mode='rb') as f:
    graph = pickle.load(f)
def get_node_embedding(node):
    embeddings = model.encode(node)
    return embeddings

def get_corrupted_embedding(word, corruption_type='misspelled'):
    if corruption_type == 'misspelled':
        corrupted_word = keyboard_typo(word)
    else:  # semantic corruption
        corrupted_word = get_closest_word(glove_vectors, word)
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

def get_closest_word(glove_vectors, word):
    similar_words = glove_vectors.most_similar(word)
    closest_word = similar_words[0][0] if similar_words else word
    return closest_word
def get_sentence_in_email(word,email):
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


#geometric_data_object = GeometricDataObject(graph)
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

def find_similar_word(word):
    try:
        most_similar = glove_vectors.most_similar(word.lower(), topn=1)
        return most_similar[0][0]
    except KeyError:
        # If the word isn't in the vocabulary, return the original word
        return word
def create_corrupted_graphs(graph):
    # Create deep copies of the original graph
    typo_graph = copy.deepcopy(graph)
    similar_word_graph = copy.deepcopy(graph)

    for email in email_nodes:
        email_edges = graph.edges(email, data=True)
        for edge in email_edges:
            source, target, attr = edge
            if graph.nodes[source]['node_type'] == 'email' and graph.nodes[target]['node_type'] == 'noun' and attr['edge_type'] == 'belongs_to':
                noun_word = target  # Assuming 'target' is the noun


                # Replace with misspelled version
                misspelled_embedding, misspelled_word = get_corrupted_embedding(noun_word, 'misspelled')
                graph.add_node(misspelled_word, node_type='noun', embedding=misspelled_embedding)

                # Replace with semantic corruption
                similar_embedding, similar_word = get_corrupted_embedding(noun_word, 'semantic')
                graph.add_node(similar_word, node_type='noun', embedding=similar_embedding)

                # Add edges with corrupt nouns
                graph.add_edge(source, misspelled_word, edge_type='belongs_to')
                graph.add_edge(source, similar_word, edge_type='belongs_to')

    return typo_graph, similar_word_graph

typo_graph, similar_word_graph = create_corrupted_graphs()

print(graph)
