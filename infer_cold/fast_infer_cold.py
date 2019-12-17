import sys
import time
import gc
import math 
import json 
import numpy as np
import numba

numba.jit(nopython=True)
def fast_neg_table_builder(word_count, table_size):
    """
    Build a negative sampling table for fast sampling under smoothed sampling probability.
    The word_count_scale variable acts like the weight in the weighted sampling.
    This function is for-loop intensive, so numba is used to accelerate.
    """
    word_count_scale = word_count.copy()
    for i in range(word_count.shape[0]):
        word_count_scale[i] = word_count_scale[i] ** 0.5
        
    neg_sample_table = []
    # init
    print('Start build the neg table')
    z = word_count_scale.sum()
    for i in range(len(word_count_scale)):
        c = word_count_scale[i]
        j = 0
        while j < int(c * table_size / z):
            neg_sample_table.append(i)
            j += 1
    return np.array(neg_sample_table, np.int64)


@numba.njit(fastmath=True)
def np_dot(x, y):
    s = 0
    for i in range(x.shape[0]):
        s += x[i] * y[i]
    return s


spec = [
    ('_neg_sample_table', numba.int64[:]),
    ('neg_table_size', numba.int64),
    ('max_sigmoid', numba.int64),          # an array field
    ('table_size', numba.int64),
    ('sigmoid_table', numba.float32[:]),
    ('_pos', numba.int64),          # an array field
    ('word_vec_np', numba.float32[:, :])
]

@numba.jitclass(spec)
class Estimator(object):
    def __init__(self, neg_table, neg_table_size, max_sigmoid, table_size, word_vec_np):
        self._neg_sample_table = neg_table
        self.neg_table_size = neg_table_size
        self.max_sigmoid = max_sigmoid
        self.table_size = table_size + 1
        self.sigmoid_table = np.zeros(self.table_size, dtype=np.float32)
        self._pos = 0
        self.word_vec_np = word_vec_np
         
        self._shuffle()
        for i in range(self.table_size):
            x = (i * 2 * self.max_sigmoid) / self.table_size - self.max_sigmoid
            self.sigmoid_table[i] = 1.0 / (1.0 + math.exp(-x))
        
    def _shuffle(self):
        np.random.shuffle(self._neg_sample_table)
    
    def sample(self):
        idx = self._neg_sample_table[self._pos]
        self._pos += 1
        need_shuffle = False
        if self._pos >= self.neg_table_size:
            self._pos = 0
            need_shuffle = True
        if need_shuffle:
            self._shuffle()
        return idx
    
    def sigmoid(self, x):
        if x < -self.max_sigmoid:
            return 0.0
        elif x > self.max_sigmoid:
            return 1.0
        else:
            idx = int((x + self.max_sigmoid) * self.table_size / self.max_sigmoid / 2.0)
            return self.sigmoid_table[idx]

    def fit_item_in_vec(self, item_word, num_iters=200, lr=0.01, min_lr=0.0001, num_neg=50, feat_dim=100):
        word_vec = np.zeros_like(self.word_vec_np[0, :], dtype=np.float32)
        grad_vec = np.zeros_like(word_vec, dtype=np.float32)
        for i in range(num_iters):
            curr_lr = min_lr + (lr - min_lr) * (1 - i / num_iters)
            grad_vec *= 0
            curr_idx = 0
            for word_idx in item_word:   
                for j in range(num_neg + 1):
                    if j == 0:
                        curr_idx = word_idx
                        label = 1.0
                    else:
                        neg_idx = self.sample()
                        curr_idx = neg_idx
                        label = 0.0
                    raw_score = np_dot(self.word_vec_np[curr_idx, :], word_vec)
                    score = self.sigmoid(raw_score)
                    alpha = curr_lr * (label - score)
                    grad_vec += self.word_vec_np[curr_idx, :] * alpha
            word_vec += grad_vec
        return word_vec 