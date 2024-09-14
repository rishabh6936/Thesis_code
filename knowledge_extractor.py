import numpy as np
import faiss
from gensim.parsing.preprocessing import STOPWORDS
from sentence_transformers import SentenceTransformer
from nltk import RegexpTokenizer
import re
import torch
import spacy

class KnowExtract:
    def __init__(self, text, trie, source_type=None):
        self.trie = trie
        self.text = text
        self.data = {}
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = self.model.to('cpu')
        self.model.to(torch.device('cpu'))
        self.source_type = source_type
        self.stopwords = None
        self.data['edges_before'] = []
        self.data['edges_after'] = []
        self.context_node_embedding = None
        self.entities = None
        self.all_trippels = None
        self.trippels = None
        self.neighborhood = None
        self.graph_edges = None
        self.nodes_embedding = None
        self.set_stopwords()
        self.set_context_node_embedding()
        self.set_entities()
        self.normalize_sentence()
        self.pos_tagging()
        self.init_data()
        self.add_occurrence_nodes()

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

    def add_occurrence_nodes(self):
            tail = np.char.array(self.entities)
            l = len(tail)
            edges_before = []
            edges_after =[]
            occurrence_nodes_before = []
            occurrence_nodes_after = []
            entity_positions = []
            entity_pos_array = []
            sentence_pattern = r'([^.?!]*[.?!])'
            sentences = re.findall(sentence_pattern, self.text)
            # Initialize a dictionary to store the presence of each entity in the text

            #find the indexes where all the entitities are present
            entity_pos_array = self.extract_entity_indexes()
            entity_pos_array = sorted(entity_pos_array)

            #find the index of all non entity indexes
            non_entity_pos_array = self.extract_non_entity_indexes(entity_pos_array)    #

            relation_before = np.char.array("comes_before")
            relation_after = np.char.array("comes_after")

            for entity in self.entities:
                single_entity_indexes = self.extract_single_entity_index(entity)
                nodes_before, nodes_after = self.find_near_index(non_entity_pos_array,single_entity_indexes)
                nodes_before_string = self.extract_non_entity_strings(nodes_before)
                nodes_after_string = self.extract_non_entity_strings(nodes_after)
                edges_before_string = [nodes_before_string, entity, relation_before]
                edges_after_string = [nodes_after_string, entity, relation_after]
                edges_before.append(edges_before_string)
                edges_after.append(edges_after_string)

            """edges_before = np.char.array(edges_before)
            edges_after = np.char.array(edges_after)"""

            """non_entity_strings = self.extract_non_entity_strings(non_entity_pos_array)

            relation_before = np.char.array("comes_before")
            relation_before = np.resize(relation_before, (l))

            relation_after = np.char.array("comes_after")
            relation_after = np.resize(relation_after, (l))


            occurrence_nodes_before = non_entity_strings[:-1]
            occurrence_nodes_after = non_entity_strings[1:]

            edges_before = np.char.array([occurrence_nodes_before, tail, relation_before]).T

            edges_after = np.char.array([occurrence_nodes_after, tail, relation_after]).T"""

            for e in edges_before:
                if e[0] != 'Empty node' and e[1] != 'Empty node':
                    self.data['edges_before'].append(e)

            for e in edges_after:
                if e[0] != 'Empty node' and e[1] != 'Empty node':
                    self.data['edges_after'].append(e)


            """sentences_before = re.findall(sentence_pattern, before_entity)
            sentence_before = sentences_before[-1].strip() if sentences_before else None
            if sentence_before == [] or sentence_before is None:
                    sentence_before = 'Empty node'

                # Find the sentence after the entity
            sentences_after = re.findall(sentence_pattern, after_entity)                sentence_after = sentences_after[0].strip() if sentences_after else None

            if sentence_after == [] or sentence_after is None:
                    sentence_after = 'Empty node'

            occurrence_nodes_before.append(sentence_before)
            occurrence_nodes_after.append(sentence_after)

            relation_before = np.char.array("comes before")
            relation_before = np.resize(relation_before, (l))
            #
            edges_before = np.char.array([occurrence_nodes_before, tail, relation_before]).T

            relation_after = np.char.array("comes after")
            relation_after = np.resize(relation_after, (l))

            edges_after = np.char.array([occurrence_nodes_after, tail, relation_after]).T

            for e in edges_before:
                if e[0] != 'Empty node' and e[1] != 'Empty node':
                    self.data['edges_before'].append(e)

            for e in edges_after:
                if e[0] != 'Empty node' and e[1] != 'Empty node':
                    self.data['edges_after'].append(e)"""

    def normalize_sentence(self):
        nlp = spacy.load('de_core_news_sm')
        # Process the sentence
        doc = nlp(self.text)

        # Extract lemmatized tokens, removing stop words and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        # Join tokens back into a single string
        normalized_sentence = ' '.join(tokens)

        return normalized_sentence

    def find_near_index(self,non_entity_pos_array,single_entity_indexes):
        nodes_before = []
        nodes_after = []
        non_entity_pos_array = sorted(non_entity_pos_array, key=lambda x: x[0])

        less_than = None
        greater_than = None

        for start_index, end_index in single_entity_indexes:
            less_than = None
            greater_than = None
            for start, end in non_entity_pos_array:
                if end <= start_index:
                    less_than = [start, end]
                elif start >= end_index and greater_than is None:
                    greater_than = [start, end]
                    break

            nodes_before.append(less_than)
            nodes_after.append(greater_than)

        return nodes_before, nodes_after





    def extract_non_entity_strings(self, non_entity_indexes):
        # Initialize a list to hold the non-entity strings
        non_entity_strings = []

        non_entity_indexes = [item if item is not None else [0, 0] for item in non_entity_indexes]

        # Iterate over the list of non-entity indexes and extract the corresponding substrings
        if non_entity_indexes is []:
            print("No non_entity")
        for start, end in non_entity_indexes:
            if start == 0 and end == 0:
                non_entity_strings.append('Empty_node')
            else:
                non_entity_strings.append(self.text[start:end])

        return non_entity_strings

    def extract_entity_indexes(self):
        entity_pos_array = []
        unique_entities = []
        sentences = self.text

        # Iterate over the entities in the original order
        for entity in self.entities:
            if entity not in unique_entities:
                unique_entities.append(entity)
        for entity in unique_entities:
                start_index = 0  # Start at the beginning of the sentence

                while start_index < len(sentences):
                    # Find the index of the next occurrence of the entity in the sentence
                    start_index = sentences.find(entity, start_index)

                    if start_index == -1:  # No more occurrences found
                        break

                    end_index = start_index + len(entity)
                    entity_positions = [start_index, end_index]
                    entity_pos_array.append(entity_positions)

                    # Move the start index forward to search for the next occurrence
                    start_index = end_index  # Move past the current found entity

        return entity_pos_array

    def extract_single_entity_index(self,entity):
        sentences = self.text
        start_index = 0  # Start at the beginning of the sentence
        entity_pos_array = []

        while start_index < len(sentences):
            # Find the index of the next occurrence of the entity in the sentence
            start_index = sentences.find(entity, start_index)

            if start_index == -1:  # No more occurrences found
                break

            end_index = start_index + len(entity)
            entity_positions = [start_index, end_index]
            entity_pos_array.append(entity_positions)

            # Move the start index forward to search for the next occurrence
            start_index = end_index  # Move past the current found entity

        return entity_pos_array



    def extract_non_entity_indexes(self, entity_indexes):
        # Initialize a list to hold non-entity indexes
        text_length = len(self.text)
        non_entity_indexes = []

        # Sort the entity indexes just in case they are not in order
        entity_indexes = sorted(entity_indexes)

        # Track the end of the last entity to find the start of the next non-entity
        last_end = 0

        for start, end in entity_indexes:
            # If there's a gap between the last end and the current start, it's a non-entity
            if start > last_end and start - last_end > 1 and start > last_end:
                non_entity_indexes.append([last_end, start])
            elif start > last_end and start - last_end == 1 and start > last_end:
                non_entity_indexes.append([0,0])

            # Update last_end to the end of the current entity
            last_end = end

        # Check for any non-entity text after the last entity
        if last_end < text_length:
            non_entity_indexes.append([last_end, text_length])



        return non_entity_indexes

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
#        tokenizer = RegexpTokenizer(r"\w+")
#        text_tokenized = tokenizer.tokenize(self.text)
        tokenizer = RegexpTokenizer(r'\d+,\d+[a-zA-Z]|\w+')
        text_tokenized = tokenizer.tokenize(self.text)
#        text_tokenized = self.text.split()
        entities = [word for word in text_tokenized if not word.lower() in self.stopwords]
        self.entities = entities

    def pos_tagging(self):
        nlp_de = spacy.load('de_core_news_sm')
        filtered_entities = []
        for entity in self.entities:
            doc = nlp_de(str(entity))
            # Keep the entity only if all tokens are nouns
            if all(token.pos_ == 'NOUN' for token in doc):
                filtered_entities.append(entity)

        self.entities = filtered_entities

    def normalize_nodes(self):
        nlp = spacy.load('de_core_news_sm')
        filtered_entities = []
        for entity in self.entities:
            doc = nlp(entity)
            for token in doc:
                lemma = token.lemma_
                filtered_entities.append(lemma)
        self.entities = filtered_entities


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
            if '/' in fact:
                split_word = fact.split('/')
                if len(split_word) > 3:  # Ensure there are at least 4 parts
                    fact = split_word[3]  # Access the fourth part (index 3)
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