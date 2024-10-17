import torch
import re
import os
from collections import Counter

import spacy
# 记得先 python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_punct and not token.is_space]


def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for src, tgt in dataset:
        counter.update(tokenize(src))
        counter.update(tokenize(tgt))

    specials = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    vocab = specials + sorted(vocab)
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word

def build_vocab_plain(text, min_freq=1):
    counter = Counter()
    counter.update(tokenize(text))
    
    specials = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    vocab = specials + sorted(vocab)
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word

if __name__ == '__main__':
    data_dir = './data'
    wiki_file = os.path.join(data_dir, 'wiki.txt')
    os.makedirs('save', exist_ok=True)
    save_dir = './save'
    wiki_vocab_file = os.path.join(save_dir, 'wiki_vocab.pt')

    with open(wiki_file, 'r') as f:
        text = f.read()

    wiki_vocab, wiki_idx2word = build_vocab_plain(text)

    torch.save(wiki_vocab, wiki_vocab_file)

