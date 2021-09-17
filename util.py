import codecs

import pandas as pd
import preprocessor as p
from nltk import tokenize
from nltk.corpus import stopwords


def read_corpus(data_df: pd) -> pd:
    data = pd.read_csv(data_df)
    return data


def get_labels(data: pd) -> list:
    return data['toxic'].values


def preprocess(sentence: str) -> str:
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)
    return p.clean(sentence)


def normalizer(sentence: str, norm) -> str:
    return norm.normalise(sentence)


def get_tokens(sentence: str) -> list:
    tokens = tokenize.word_tokenize(sentence, language='portuguese')
    return [t.lower() for t in tokens if t not in stopwords.words(u'portuguese')]


def create_preprocessed_sentences(dev_sentences):
    with open('data/prep-test.txt', 'w') as f:
        for snt in dev_sentences:
            f.write(snt + '\n')
    exit()


def get_tokenized_sentences(data_train: pd, data_dev: pd) -> (list, list):
    train_sentences, dev_sentences = [], []
    for tweet in data_train['text']:
        train_sentences.append(''.join(get_tokens(tweet)))
    for tweet in data_dev['text']:
        dev_sentences.append(''.join(get_tokens(tweet)))
    return train_sentences, dev_sentences


def get_sentences(data_dev: pd, norm) -> (list, list):
    train_sentences, dev_sentences = [], []
    # for tweet in data_train['text']:
    #     train_sentences.append(normalizer(preprocess(tweet), norm))
    for tweet in data_dev['text']:
        dev_sentences.append(normalizer(preprocess(tweet), norm))
    create_preprocessed_sentences(train_sentences, dev_sentences)
    return train_sentences, dev_sentences


def update_csv(data: pd):
    lines = []
    with open('data/prep-test.txt') as f:
        for line in f.readlines():
            lines.append(line.strip())
    for idx, tweet in enumerate(data['text']):
        data._set_value(idx, 'text', lines[idx])
    data.to_csv('test.csv', index=False)


if __name__ == '__main__':
    corpus = pd.read_csv('data/ToLD-BR.csv')
    with codecs.open('prep/example.txt', 'w', 'utf-8') as f:
        for id, label, text in zip(corpus['i'], corpus['toxic'], corpus['text']):
            if label == '1':
                f.write(str(id) + '\t' + 'toxic' + '\t' + text + '\n')
            else:
                f.write(str(id) + '\t' + 'non-toxic' + '\t' + text + '\n')

