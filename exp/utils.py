import os
import pickle
import nltk
import numpy as np
import pandas as pd


class ProdParser:
    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()

    def process(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tokens = [self.wnl.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t.isalpha()]
        return tokens


class EntityIndexer:
    def __init__(self, name):
        self.entity2ind = {}
        self.default_path = '{}_info_index.pkl'.format(name)

    def _index(self, t):
        if t not in self.entity2ind:
            self.entity2ind[t] = len(self.entity2ind)
        return self.entity2ind[t]

    def index(self, tokens):
        if isinstance(tokens, list):
            return [self._index(t) for t in tokens]
        elif isinstance(tokens, str) or isinstance(tokens, int):
            return self._index(tokens)
        else:
            raise ValueError(type(tokens))

    def save(self, path=None):
        if path is None:
            path = self.default_path
        with open(path, 'wb') as file:
            pickle.dump(self.entity2ind, file)

    def load(self, path=None):
        if path is None:
            path = self.default_path
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'rb') as file:
            self.entity2ind = pickle.load(file)


class VecStore:
    def __init__(self, prefix, init=True):
        self.prefix = prefix
        self.item_in = None
        self.item_i_out = None
        self.item_u_out = None
        self.user_in = None
        self.word_out = None

        if init:
            self.load()
            self.normalize()

    def load(self):
        self.item_in = np.load('../output/{}.vec_itemInput.vec.npy'.format(self.prefix))
        item_out = np.load('../output/{}.vec_itemOutput.vec.npy'.format(self.prefix))
        self.user_in = np.load('../output/{}.vec_userInput.vec.npy'.format(self.prefix))
        self.word_out = np.load('../output/{}.vec_wordOutput.vec.npy'.format(self.prefix))

        if item_out.shape[1] != self.item_in.shape[1]:
            self.item_i_out = item_out[:, self.user_in.shape[1]:]
            self.item_u_out = item_out[:, :self.user_in.shape[1]]
        else:
            self.item_i_out = item_out
            self.item_u_out = item_out

    @staticmethod
    def _normalize(vec):
        return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]

    def normalize(self):
        self.item_in = VecStore._normalize(self.item_in)
        self.item_i_out = VecStore._normalize(self.item_i_out)


class SearchEngine:
    def __init__(self, vec_store, index):
        self.vec_store = vec_store
        self.prod_idx2id = {v:k for k, v in index.entity2ind.items()}
        self.prod_id2idx = index.entity2ind
        self.item_info = pd.read_csv('../data/instacart/products.csv')
        self.item_info.set_index('product_id', inplace=True)

    def find_sim(self, prod_id, topk=10):
        prod_idx = self.prod_id2idx[prod_id]
        sim_prod_idxes = self.vec_store.item_in[prod_idx].dot(self.vec_store.item_in.T).argsort()[::-1][:topk]
        return [self.prod_idx2id[x] for x in sim_prod_idxes]

    def display_sim(self, prod_id, topk=10):
        sim_prod_ids = self.find_sim(prod_id)
        return self.item_info.loc[[prod_id] + sim_prod_ids]['product_name']

    def find_comp(self, prod_id, topk=10):
        prod_idx = self.prod_id2idx[prod_id]
        comp_prod_idxes = self.vec_store.item_in[prod_idx].dot(self.vec_store.item_i_out.T).argsort()[::-1][:topk]
        return [self.prod_idx2id[x] for x in comp_prod_idxes]

    def display_comp(self, prod_id, topk=10):
        comp_prod_ids = self.find_comp(prod_id)
        return self.item_info.loc[[prod_id] + comp_prod_ids]['product_name']