import numpy as np
from gensim.models import KeyedVectors


class Embedding:

    def __init__(self, general_model):
        self.general_model = KeyedVectors.load(general_model, mmap='r')

    def get_embeddings(self, token, sentence):
        try:
            return np.mean(self.general_model[token], axis=0)
            # return np.mean(self.hate_model.get_vector(token))
        except KeyError:
            try:
                return np.mean(self.general_model[token], axis=0)
            except KeyError:
                embedded = []
                for snt in sentence:
                    try:
                        embedded.append(np.mean(self.general_model[snt]))
                        # embedded.append(np.mean(self.hate_model.get_vector(snt)))
                    except KeyError:
                        embedded.append(0.0)
                return np.mean(embedded)
