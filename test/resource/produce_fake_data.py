import argparse
import numpy as np 
import pandas as pd

def main():
    #genearte random dict for item
    item2word = []
    c = 0
    for i in range(args.n_item):
        s = [i]
        for _ in range(10):
            s.append(c)
            c += 1
            c = c % args.n_word
        item2word.append(s)

    d = pd.DataFrame(item2word)
    d.to_csv('fake_item_word.txt', index=False, header=None, sep='\t')


    #genearte random dict for user
    item2word = []
    c = 0
    for i in range(args.n_user):
        s = [i]
        for _ in range(10):
            s.append(c)
            c += 1
            c = c % args.n_user_word
        item2word.append(s)

    d = pd.DataFrame(item2word)
    d.to_csv('fake_user_word.txt', index=False, header=None, sep='\t')


    # generate random user trx history
    user = 0
    item = 0
    obsHist = []
    with open('fake_user_hist.txt', 'w') as f:
        fake_timestamp_str = ','.join([str(i + 0.1) for i in range(1, args.seq_len + 1)])
        for i in range(args.n_obs):
            s = [str(user), fake_timestamp_str]
            user += 1
            user = user % args.n_user

            fake_item_l = []
            for _ in range(args.seq_len):
                item += 1
                item = item % args.n_item
                fake_item_l.append(str(item))
            s.append(','.join(fake_item_l))
            s_str = '\t'.join(s)
            f.write(s_str + '\n')

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Produce fake data for debug only')
    parser.add_argument('--n_user', type=int, default=100)
    parser.add_argument('--n_item', type=int, default=5000)
    parser.add_argument('--n_word', type=int, default=500)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_user_word', type=int, default=500)
    parser.add_argument('--n_obs', type=int, default=1000)

    args = parser.parse_args()

    main()