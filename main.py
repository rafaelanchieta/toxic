import numpy
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

import util
from generate_graph import GenerateGraph
from regularization import Regularization

if __name__ == '__main__':
    test_corpus = 'prep/test.csv'
    train_corpus = 'prep/train.csv'
    test_data = util.read_corpus(test_corpus)
    train_data = util.read_corpus(train_corpus)

    graph, node_sentence = GenerateGraph(train_data, test_data).generate_graph()

    train_labels = util.get_labels(train_data)
    dev_labels = util.get_labels(test_data)
    reg = Regularization().regularizer(graph, node_sentence, train_labels, 'features', pre_annotated=0.3, method='gfhf')
    train = pd.read_csv('features/train.csv')
    dev = pd.read_csv('features/test.csv')

    clf = GradientBoostingClassifier(n_estimators=500, max_depth=5)

    clf.fit(train[['toxic', 'nontoxic']], numpy.ravel(train[['label']]))
    # print('Training: ', clf.score(dev[['toxic', 'nontoxic']], dev_labels))
    y_pred = clf.predict(dev[['toxic', 'nontoxic']])

    print(classification_report(dev_labels, y_pred))
