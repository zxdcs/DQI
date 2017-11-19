import pickle
import os
import spacy
import numpy
import csv

nlp = spacy.load('en')


def read_all_data(file='data/quora_duplicate_questions.tsv'):
    with open(file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = list(reader)
        data = numpy.asarray(data)
        numpy.random.seed(123)
        numpy.random.shuffle(data)
        length = data.shape[0]
        train = data[:int(0.8 * length)]
        valid = data[int(0.8 * length):int(0.9 * length)]
        test = data[int(0.9 * length):]
        return train, valid, test


def tokenize_data(all_data):
    tokenized = []
    for set_ in all_data:
        data = {'q1': [], 'q2': [], 'y': []}
        for datum in set_:
            data['q1'].append(_text_preprocess(datum['question1']))
            data['q2'].append(_text_preprocess(datum['question2']))
            data['y'].append(int(datum['is_duplicate']))
        tokenized.append(data)
    return tokenized


def _text_preprocess(text):
    if text is None:
        return []
    text = text.strip().replace('`', "'")
    doc = nlp.tokenizer(text)
    tokens = [t.lower_ for t in doc]
    return tokens


def _read_emb(file, dim):
    emb = {}
    dim += 1
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if len(tokens) == dim:
                emb[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))
    return emb


def _token2idx(tokens, token_map, embs, filtered_emb):
    for i in range(len(tokens)):
        if tokens[i] not in token_map:
            if tokens[i] in embs:
                token_map[tokens[i]] = len(token_map)
                filtered_emb.append(embs[tokens[i]])
            else:
                tokens[i] = '<unk>'
        tokens[i] = token_map[tokens[i]]


def idx_and_emb(all_data, emb_file, dim):
    embs = _read_emb(emb_file, dim)
    word2idx = {'<pad>': 0, '<unk>': 1}
    filtered_emb = [numpy.random.uniform(-0.1, 0.1, dim) for _ in range(2)]
    for set_ in all_data:
        for datum in set_['q1']:
            _token2idx(datum, word2idx, embs, filtered_emb)
        for datum in set_['q2']:
            _token2idx(datum, word2idx, embs, filtered_emb)
    print('{} word types'.format(len(word2idx)))
    filtered_emb = numpy.asarray(filtered_emb, dtype='float32')
    return filtered_emb, word2idx


def main():
    all_data = read_all_data()
    print('Reading data done.')
    tokenized = tokenize_data(all_data)
    print('Tokenization done.')
    emb_file = os.path.join(os.path.expanduser('~'), 'Data/Embeddings/glove.840B.300d.txt')
    filtered_emb, word2idx = idx_and_emb(tokenized, emb_file, 300)
    print('Embedding done.')
    with open('data/data_emb', 'wb') as f:
        pickle.dump((tokenized, filtered_emb, word2idx), f)
    print('Saved.')


if __name__ == '__main__':
    main()
