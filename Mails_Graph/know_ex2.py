import numpy as np
import faiss
from gensim.parsing.preprocessing import STOPWORDS
from sentence_transformers import SentenceTransformer
from nltk import RegexpTokenizer
import re
import torch

class KnowExtract:
    def __init__(self, text, trie, source_type=None):
        self.trie = trie
        self.text = text
        self.data = {}
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#        self.model = SentenceTransformer('all-MiniLM-L6-v2')
#        self.model = self.model.to('cpu')
        self.model.to(torch.device('cpu'))
        self.context_node_embedding = None
#        self.set_stopwords()
        self.set_context_node_embedding()


    def init_data(self):
        tail = np.char.array(self.entities)
        l = len(tail)
        head = np.char.array(self.text)
        head = np.resize(head, (l))

        relation = np.char.array("belongs to")
        relation = np.resize(relation, (l))
        edges = np.char.array([head, tail, relation]).T
        self.data['msg_knowledge_edges'] = edges
        self.data['knowledge_nodes'] = tail

    def add_occurence_nodes(self):
        tail = np.char.array(self.entities)
        l = len(tail)
        self.data['occurence_nodes_before'] = []
        self.data['occurence_nodes_after'] = []
        for entity in tail:
            entity_index = self.text.find(entity)
            if entity_index == -1:
                return None, None  # Entity not found in the text

            before_entity = self.text[:entity_index].strip()
            after_entity = self.text[entity_index:].strip()

            sentence_pattern = r'([^.?!]*[.?!])'

            # Find the sentence before the entity
            sentences_before = re.findall(sentence_pattern, before_entity)
            sentence_before = sentences_before[-1].strip() if sentences_before else None
            if sentence_before == [] or sentence_before is None:
                sentence_before = 'Empty node'

            # Find the sentence after the entity
            sentences_after = re.findall(sentence_pattern, after_entity)
            sentence_after = sentences_after[0].strip() if sentences_after else None

            if sentence_after == [] or sentence_after is None:
                sentence_after = 'Empty node'

            self.data['occurence_nodes_before'].append(sentence_before)
            self.data['occurence_nodes_after'].append(sentence_after)

        relation_before = np.char.array("mentioned in")
        relation_before = np.resize(relation_before, (l))

        edges_before = np.char.array([self.data['occurence_nodes_before'], tail, relation_before]).T

        relation_after = np.char.array("comes before")
        relation_after = np.resize(relation_after, (l))

        edges_after = np.char.array([self.data['occurence_nodes_after'], tail, relation_after]).T


        #return sentence_before, sentence_after

    def get_data(self, key):
        return self.data[key]

    def get_graph_edges(self):
        return self.graph_edges

    def get_trippels(self):
        return self.trippels

    def set_stopwords(self):
        german_pronouns = set(['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'der', 'die', 'das'])  # German equivalents
        english_pronouns = set(['I', 'you', 'he', 'she', 'it', 'we', 'they'])

        # Set of integers from 0 to 9 as strings
        integer_stopwords = set([str(i) for i in range(1000)])

        # Combine all stopwords
        self.stopwords = STOPWORDS.union(english_pronouns).union(german_pronouns).union(integer_stopwords)

    def get_stopwords(self):
        return self.stopwords

    def set_context_node_embedding(self):

       self.context_node_embedding = self.model.encode([self.text])
#        x = self.model.encode('Hello')
       self.context_node_embedding = torch.tensor(self.context_node_embedding)
       self.context_node_embedding = self.context_node_embedding.to("cpu")

    def get_text_embedding(self):
        return self.text_embedding

    def get_neighborhood(self):
        return self.neighborhood

    def set_entities(self):
        tokenizer = RegexpTokenizer(r"\w+")
        text_tokenized = tokenizer.tokenize(self.text)
        entities = [word for word in text_tokenized if not word.lower() in self.stopwords]
        self.entities = entities

    def get_entities(self):
        return self.entities

    def check_emb(self, emb):
        if type(emb) != np.ndarray:
            emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb

    def search_neighborhood(self, k):
        if type(self.trippels) == type(None):
            return
        # extract the doublet trippels
        tuppel = np.char.array([self.trippels[:, 0], self.trippels[:, 1]]).T
        cleared_tuppel, tuppel_indicies = np.unique(tuppel, return_index=True, axis=0)
        self.trippels = self.trippels[tuppel_indicies]
        self.nodes_embedding = self.nodes_embedding[tuppel_indicies]

        # reduce k
        if len(self.trippels) < k:
            # extract best trippels
            self.neighborhood = self.trippels

        else:
            # search for best trippels
            # cpu search
            d = self.context_node_embedding.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(self.nodes_embedding)
            D, I = index.search(self.context_node_embedding, k)

            # extract best trippels
            self.neighborhood = self.trippels[I[0]]

            # delete extracted trippels
            mask = np.ones(len(self.trippels), dtype=bool)
            mask[I[0]] = False
            self.trippels = self.trippels[mask]
            self.nodes_embedding = self.nodes_embedding[mask]

    def new_hop(self, hop_nr, k=100):
        if type(self.neighborhood) != type(None):
            self.entities = np.concatenate([self.entities, self.neighborhood[:, 1]])
            self.pull_from_conceptnet_trie(facts=self.neighborhood[:, 1])
        else:
            self.pull_from_conceptnet_trie()

        self.search_neighborhood(k=k)

        if type(self.graph_edges) == type(None):
            self.graph_edges = self.neighborhood
        else:
            self.graph_edges = np.concatenate(
                (self.graph_edges, self.neighborhood), axis=0)

        if type(self.graph_edges) == type(None):
            return 0
        return len(self.graph_edges)

    def pull_from_conceptnet_trie(self, facts=None):
        if type(facts) == type(None):
            facts = self.entities

        if type(facts) == type(""):
            facts = [facts]

        trippels = []
        emb = []
        for fact in facts:
            fact = fact.lower()
            trippel, embedding = self.trie.query(fact)
            trippels += trippel
            emb += embedding

        if len(trippels) > 0:
            if type(self.trippels) == type(None):
                self.trippels = np.char.array(trippels)
            else:
                self.trippels = np.concatenate((self.trippels, trippels), axis=0)

            if type(self.nodes_embedding) == type(None):
                self.nodes_embedding = self.check_emb(emb)
            else:
                self.nodes_embedding = np.concatenate((self.nodes_embedding, self.check_emb(emb)), axis=0)