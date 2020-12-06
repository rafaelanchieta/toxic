from random import shuffle

import pandas as pd

from GNetMine import GNetMine
from gfhf import harmonic_function
from llgc import local_and_global_consistency


class Regularization:

    def __init__(self):
        pass

    @staticmethod
    def regularizer(graph, sentence_nodes, train_labels, path_out, pre_annotated, method):
        total_samples = len(train_labels)
        cods = list(range(total_samples))
        shuffle(cods)

        annotated = cods[0:int(total_samples * pre_annotated)]
        train = 'train_%d'
        # test = 'test_%d'

        key_annotated = []
        for i in range(total_samples):
            if i in annotated:
                graph.nodes[sentence_nodes[train % i]]['label'] = train_labels[i]
                key_annotated.append(train % i)
        with open('annotated.txt', 'w') as f:
            for i in annotated:
                f.write('%d\n' %i)

        labels = ['toxic', 'nontoxic']
        columns = ['id', labels[0], labels[1], 'label']
        rows_train, rows_test = [], []

        if method == 'gfhf':
            f = harmonic_function(graph)
        elif method == 'llgc':
            f = local_and_global_consistency(graph)

        if method in ['gfhf', 'llgc']:
            for key in sentence_nodes.keys():
                id_node = sentence_nodes[key]
                split_key_node = key.split('_')
                t = split_key_node[0]
                cod = int(split_key_node[1])
                if t == 'train':
                    rows_train.append([id_node, f[id_node][0], f[id_node][1], train_labels[cod]])
                elif t == 'dev':
                    rows_test.append([id_node, f[id_node][0], f[id_node][1]])
        elif method == 'gnetmine':
            m = GNetMine(graph=graph)
            m.run()
            f = m.f['sentence']
            # labels = m.labels
            nodes = m.nodes_type['sentence']
            dict_nodes = {k: i for i, k in enumerate(nodes)}
            for key in sentence_nodes.keys():
                id_node = sentence_nodes[key]
                split_key_node = key.split('_')
                t = split_key_node[0]
                cod = int(split_key_node[1])
                if t == 'train':
                    rows_train.append([id_node, f[dict_nodes[id_node]][0], f[dict_nodes[id_node]][1], train_labels[cod]])
                elif t == 'dev':
                    rows_test.append([id_node, f[dict_nodes[id_node]][0], f[dict_nodes[id_node]][1]])

        file_train = 'features_%s_pre_annotated_train.csv' % len(annotated)
        file_test = 'features_%s_pre_annotated_test.csv' % len(annotated)

        df_train = pd.DataFrame(rows_train, columns=columns)
        df_test = pd.DataFrame(rows_test, columns=columns[:3])
        df_train.to_csv(path_out + file_train, index=False)
        df_test.to_csv(path_out + file_test, index=False)