import networkx

import util


class GenerateGraph:

    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data
        self.graph = networkx.Graph()
        self.dic_token_nodes = {}
        self.node_sentence = {}
        self.node_ids = 0

    def create_graph(self, split_data, split_name):
        for idx, sentence in enumerate(split_data):
            key = split_name + '_%d' % idx
            self.node_sentence[key] = self.node_ids
            self.node_ids += 1
            self.graph.add_node(self.node_sentence[key], type='sentence', value=key)

            for token in sentence:
                if token not in self.dic_token_nodes:
                    self.dic_token_nodes[token] = self.node_ids
                    self.node_ids += 1
                    self.graph.add_node(self.dic_token_nodes[token], type='token', value=token)
                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[token])
                else:
                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[token])

    def generate_graph(self):
        train_sentences, dev_sentences = util.get_tokenized_sentences(self.train_data, self.dev_data)
        self.create_graph(train_sentences, 'train')
        self.create_graph(dev_sentences, 'dev')
        return self.graph, self.node_sentence
