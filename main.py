import numpy
import pandas
from enelvo import normaliser
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import util
from generate_graph import GenerateGraph
from regularization import Regularization

if __name__ == '__main__':
    test_corpus = 'prep/test.csv'
    train_corpus = 'prep/train.csv'
    test_data = util.read_corpus(test_corpus)
    train_data = util.read_corpus(train_corpus)
    # util.update_csv(dev_data)
    # exit()
    # norm = normaliser.Normaliser()
    graph, node_sentence = GenerateGraph(train_data, test_data).generate_graph()
    train_labels = util.get_labels(train_data)
    dev_labels = util.get_labels(test_data)
    reg = Regularization().regularizer(graph, node_sentence, train_labels, '', pre_annotated=0.4, method='gnetmine')
    train = pandas.read_csv('features_6720_pre_annotated_train.csv')
    dev = pandas.read_csv('features_6720_pre_annotated_test.csv')

    # clf = MLPClassifier(solver='adam', random_state=42, max_iter=1000, verbose=True)
    clf = SVC(verbose=True)
    # clf = GaussianNB()
    # clf = DecisionTreeClassifier()
    # clf = GradientBoostingClassifier()
    # clf = XGBClassifier(n_estimators=500)
    clf.fit(train[['toxic', 'nontoxic']], numpy.ravel(train[['label']]))
    print('Training: ', clf.score(dev[['toxic', 'nontoxic']], dev_labels))
    y_pred = clf.predict(dev[['toxic', 'nontoxic']])

    print(classification_report(dev_labels, y_pred))
    # print(util.normalizer(util.preprocess(test_data['text'][13])))
