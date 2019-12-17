import os
import logging
import json
import argparse
import multiprocessing
import queue
from multiprocessing import Process, Queue
from time import sleep

import numpy as np

from fast_infer_cold import fast_neg_table_builder, Estimator

SEP = '\t'
NULL = '\\N'

logging.basicConfig(level=logging.DEBUG, filename='result/log.log',
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser()

parser.add_argument('--item_word', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--num_worders', type=int, default=4)
parser.add_argument('--max_word_num', type=int, default=40)

args = parser.parse_args()

item_word_file = args.item_word 
word_out_file = args.output

NUM_WORKER=args.num_worders
MAX_WORD_NUM=args.max_word_num

def np1d2str(x):
        x = x.astype(np.float32)
        res = []
        for i in range(x.shape[0]):
            res.append(str(x[i]))
        return ','.join(res)
    
def check_null(x):
    return x is None or NULL in x or 'NULL' in x or len(x) == 0

def parse_line(line):
    line = line.strip()
    try:
        cid, word_idx_str = line.split(SEP)
    except ValueError as e:
        cid, word_idx_str = None, None

    if check_null(cid) or check_null(word_idx_str):
        return None, None

    word_idx_str = word_idx_str.replace('"', '')
    word_idx_list = [int(x) for x in word_idx_str.split(',')]
    return cid, word_idx_list

def main():
    word_out_np = np.load(word_out_file).astype(np.float32)
    word_count = np.load('word_count.npy')

    SEP = '\t'
    NULL = '\\N'

    logging.info('Loading data: Start')
    data = []
    with open(item_word_file) as f:
        cc = 0
        for line in f:
            cid, word_idx_list = parse_line(line)
            if cid is None or word_idx_list is None:
                continue
            data.append((cid, word_idx_list))
    logging.info('Loading data: Done')

    logging.info('Put data into queue: Start')
    data_queue = Queue()
    result_queue = Queue()
    for x in data:
        data_queue.put(x)
    logging.info('Put data into queue: Done')

    def infer_result(data_queue, result_queue, word_count, word_out_np):
        NEGATIVE_TABLE_SIZE = 5000000
        neg_table = fast_neg_table_builder(word_count, NEGATIVE_TABLE_SIZE)
        full_est = Estimator(neg_table, neg_table.shape[0], 8, 512, word_out_np)
        while True:
            try:
                cid, word_idx_list = data_queue.get(timeout=1.5)
                if len(word_idx_list) > MAX_WORD_NUM:
                    continue
                embeddings = full_est.fit_item_in_vec(word_idx_list)
                result_queue.put((cid, embeddings))
            except queue.Empty:
                logging.info('Process {} finished'.format(os.getpid()))
                return


    def save(file_name, result_queue):
        count = 0
        id2idx = {}
        data = []

        logging.info('Saving embeddings: Start')
        while True:
            try:
                cid, embed = result_queue.get(timeout=10)
                if cid not in id2idx:
                    id2idx[cid] = len(id2idx)
                    data.append(embed)
            except queue.Empty:
                logging.info('Result queue empty')
                break

        with open(file_name + '_id2idx.json', 'w') as f:
            json.dump(id2idx, f)

        data = np.array(data)
        np.save(file_name + '.npy', data)

        logging.info('Saving embeddings: Done')


    embedings_workder = []
    for _ in range(NUM_WORKER):
        p = Process(target=infer_result, args=(data_queue, result_queue, word_count, word_out_np))
        # p.daemon = True
        embedings_workder.append(p)

    # save_worker = Process(target=saver, args=('result/embed', queue, result_queue))
    # save_worker.daemon = True

    try:
        for p in embedings_workder:
            p.start()

        while data_queue.qsize() > 0:
            logging.info('Current left: {}, Result queue size: {}'.format(data_queue.qsize(), result_queue.qsize()))
            sleep(5)

        logging.info('Training Done')

        # for p in embedings_workder:
        #     p.join()
        save('result/embed', result_queue)
        logging.info('All Done')

    except:
        import traceback
        logging.debug(traceback.print_tb())
        for p in embedings_workder:
            p.terminate()
        for p in embedings_workder:
            p.join()
    
if __name__ == '__main__':
    main()