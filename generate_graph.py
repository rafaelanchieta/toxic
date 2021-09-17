import json

import networkx
from nltk import word_tokenize

from embedding import Embedding


class GenerateGraph:

    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data
        self.graph = networkx.Graph()
        self.dic_token_nodes = {}
        self.node_sentence = {}
        self.node_ids = 0
        self.embeddings = Embedding('embeddings/glove')
        self.vocabulary = self.read_resource('data/vocabulary.txt', 'data/badword_list.json')

    @staticmethod
    def read_resource(vocabulary: str, word_list: str) -> list:
        vocab = []
        with open(vocabulary, 'r') as f:
            for line in f.readlines():
                vocab.extend(line.split(','))
        with open(word_list, 'r') as f:
            data = json.loads(f.read())
            vocab.extend(set(list(data)))
        return vocab

    def create_graph(self, split_data, split_name):
        for idx, sentence in enumerate(split_data):
            key = split_name + '_%d' % idx
            self.node_sentence[key] = self.node_ids
            self.node_ids += 1
            self.graph.add_node(self.node_sentence[key], type='sentence', value=key)

            tokens = word_tokenize(sentence, language='portuguese')
            for tk in tokens:
                if tk not in self.dic_token_nodes:
                    self.dic_token_nodes[tk] = self.node_ids
                    self.node_ids += 1
                    self.graph.add_node(self.dic_token_nodes[tk])
                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[tk])
                    weight = float('%.4f' % (self.get_weight(tk, sentence)))

                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[tk], weight=weight)
                else:
                    weight = float('%.4f' % (self.get_weight(tk, sentence)))
                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[tk], weight=weight)

    def get_weight(self, token: str, sentence: str) -> float:
        if token in self.vocabulary:
            return self.embeddings.get_embeddings(token, sentence)
        else:
            return 0.0

    def generate_graph(self):
        self.create_graph(self.train_data['text'].values, 'train')
        self.create_graph(self.dev_data['text'].values, 'dev')
        return self.graph, self.node_sentence
