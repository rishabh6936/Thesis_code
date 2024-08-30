from trie_conceptnet import Trie
from trie_structure import Trie_hash
from knowledge_extractor import KnowExtract
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os
import pickle

class GraphBuilder:
    def __init__(self,
                 path='/Users/rishabhsingh/Shira_thesis/Crawlers/dia_trans_net/data/conceptNet_embs',
                 save_path='/Users/rishabhsingh/PycharmProjects/Mails_Graph/datasets/',
                 emb_file_name='conceptnet_embs'):
        self.path = path
        self.save_path = save_path
        self.emb_file_name = emb_file_name
        self.embeddings = None
        self.trie = None
        self.dataset = None
        self.conceptNet = None
        self.start_index = None
        self.set_dataset()
        self.set_embeddings()
        self.set_trie()


    def set_dataset(self):
            print("load and save dataset")
            dataset = load_dataset("conceptnet5")
            conceptNet = dataset.filter(lambda example: example['lang'] == 'de')
            self.conceptNet = conceptNet['train']
    def set_embeddings(self):
            print("load and save embeddings")
            self.embeddings = self.get_embeddings()

    def get_embeddings(self, numpy_array=True):
            try:
                embeddings = self.load_tensor(file_name=self.emb_file_name, path=self.path)
            except:
                print("no saved embeddings could be found")
                dataset = load_dataset("conceptnet5")
                german_dataset = dataset.filter(lambda example: example['lang'] == 'de')
#                model = SentenceTransformer('all-MiniLM-L6-v2')
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                embeddings = model.encode(german_dataset['train']['arg2'])
                self.save_tensor(embeddings, file_name=self.emb_file_name, path=self.path)
            if numpy_array:
                embeddings = embeddings.cpu().detach().numpy()
            return embeddings

    def load_tensor(self, file_name, path):
            f = os.path.join(path, f"{file_name}.pt")
            return torch.load(f)

    def save_tensor(self, tensor, file_name, path):
            if not torch.is_tensor(tensor):
                tensor = torch.from_numpy(tensor)
            f = os.path.join(path, f"{file_name}.pt")
            torch.save(tensor, f)

    def set_trie(self):
            print("build trie datastructure")

            try:
                with open('/Users/rishabhsingh/Shira_thesis/Crawlers/dia_trans_net/data/Trie/trie.pkl', 'rb') as file:
                     self.trie = pickle.load(file)

            except:
                self.trie = Trie()
                self.trie.insert_dataset(self.conceptNet, self.get_embeddings())
                with open('/Users/rishabhsingh/Shira_thesis/Crawlers/dia_trans_net/data/Trie/trie.pkl', 'wb') as file:
                    pickle.dump(self.trie, file)
                del self.conceptNet
                del self.embeddings