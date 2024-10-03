import pickle
from graph_creation import replace_misspelled_candidate
import re
import gensim.downloader


# Load pre-trained word vectors (e.g., 'glove-wiki-gigaword-100' or 'word2vec-google-news-300')
glove_vectors = gensim.downloader.load('word2vec-google-news-300')  # or 'word2vec-google-news-300'

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

save_path = '/Users/rishabhsingh/Rishabh_thesis_code/Mails_Graph/saved_data/graph.pkl'
with open(save_path, mode='rb') as f:
    graph = pickle.load(f)

email_nodes = [node for node, attr in graph.nodes(data=True) if attr.get('node_type') == 'email']

suitable_edges = []
for email in email_nodes:
    email_edges = graph.edges(email, data=True)
    for edge in email_edges:
        source, target, attr = edge
        # Determine the edge type based on source and target node types
        source_type = graph.nodes[source].get('node_type')
        target_type = graph.nodes[target].get('node_type')
        attr_type = attr['edge_type']

        if source_type == 'email' and target_type == 'noun' and attr_type == 'belongs_to':
           suitable_edges.append(edge)

for edges in suitable_edges:
    noun = suitable_edges[1]
    corrupt_word = keyboard_typo(noun)
    sentence_hit = get_sentence_in_email(noun, suitable_edges[0])

    #using this function to actually replace the misspelled word in place of correct
    replaced_sentence = replace_misspelled_candidate(noun, corrupt_word, sentence_hit)
    glove_vectors.most_similar(noun)


print(graph)
