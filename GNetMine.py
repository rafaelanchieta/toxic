import numpy as np
import networkx as nx


# from graph_util import *
# import HetClass
# from HetClass.models.HetGraph import HetGraph

# -------------------------------------------------------------------- #
class GNetMine:
    '''
    Attributes
    ----------

    '''

    # ---------------------------------------------------------------- #
    # Inicialization and auxiliar methods
    # ---------------------------------------------------------------- #
    def __init__(self, graph=None, y=None, alpha=0.1, gamma=0.2):
        self.graph = graph
        self.y = y
        self.alpha = alpha
        self.gamma = gamma
        self.R = dict()
        self.D = dict()
        self.f = list()
        self.labels = list(set(nx.get_node_attributes(self.graph, 'label').values()))  # quais são as classes
        self.types = list(set(nx.get_node_attributes(self.graph, 'type').values()))
        self.nodes_type = {i: [key for key, attr in self.graph.nodes(data=True) if attr['type'] == i] for i in
                           self.types}

    def get_relation_matrix(self, i, j):
        return self.R.get((i, j), None)

    def set(self, y=None, alpha=None, gamma=None):
        if y is not None:
            self.y = y

        if alpha is not None:
            self.alpha = alpha

        if gamma is not None:
            self.gamma = gamma

    def build_relation_matrix(self):
        # Do grafo pega as dimensoes dos objetos de tipo i e j
        graph = self.graph
        # types = list(set(nx.get_node_attributes(self.graph, 'type').values()))

        # Para todo par de objetos pergunta a ed do grafo de existe uma
        # aresta entre eles, se sim atribui o valor de 1 na matriz
        for i in self.types:
            for j in self.types:
                lista_nodes = []

                # tipoi = [key for key, attr in self.graph.nodes(data=True) if attr['type'] == i]
                tipoi = self.nodes_type[i]
                # tipoj = [key for key, attr in self.graph.nodes(data=True) if attr['type'] == j]
                tipoj = self.nodes_type[j]
                lista_nodes.extend(tipoi)
                lista_nodes.extend(tipoj)

                lista_cods = {}
                lista_cods.update({key: ind for key, ind in zip(tipoi, range(len(tipoi)))})
                lista_cods.update({key: ind for key, ind in zip(tipoj, range(len(tipoj)))})

                list_edges = nx.convert.to_dict_of_lists(self.graph, nodelist=lista_nodes)
                Rij = np.zeros([len(tipoi), len(tipoj)])

                for k in tipoi:
                    for w in list_edges[k]:
                        Rij[lista_cods[k], lista_cods[w]] = 1
                # Rij = nx.convert_matrix.to_numpy_matrix(self.graph, nodelist=lista_nodes)
                # Rij = graph.get_relation_matrix(i,j)
                if np.any(Rij):
                    self.R[(i, j)] = Rij

    def build_S(self):
        S = dict()
        for (i, j) in self.R:
            Rij = self.R[(i, j)]

            dij = np.nansum(Rij, axis=1)
            dji = np.nansum(Rij, axis=0)

            dij = 1.0 / (dij ** .5)
            dji = 1.0 / (dji ** .5)
            dij[np.isinf(dij) | np.isnan(dij)] = 0
            dji[np.isinf(dji) | np.isnan(dji)] = 0
            # print(dij.min())
            # print(dij.max())
            # print(dji.min())
            # print(dji.max())
            S[(i, j)] = np.dot(np.diag(dij), Rij)
            S[(i, j)] = np.dot(S[(i, j)], np.diag(dji))

        return S

    def initialize(self):
        # ---------------------------------------------------------------- #
        # Metodo para inicializar as variaveis auxiliares para a iteracao do
        # metodo
        # ---------------------------------------------------------------- #
        # types = list(set(nx.get_node_attributes(self.graph, 'type').values()))
        m = len(self.types)
        self.f = dict()
        self.y = dict()
        # labels = list(set(nx.get_node_attributes(self.graph, 'label').values())) # quais são as classes
        K = len(self.labels)

        for i in self.types:
            # nodes_ni = [key for key, attr in self.graph.nodes(data=True) if attr['type'] == i]
            nodes_ni = self.nodes_type[i]
            ni = len(nodes_ni)
            # ni = self.graph.get_n(i) # quantos nós são de determinado tipo, lembrar: Grafo heterogeneo
            fi = np.zeros((ni, K))

            for p, key in enumerate(nodes_ni):
                # classi = self.graph.get_class(i,p)
                # 'label' in G.nodes['C1'].keys()
                # if classi is not None:
                if 'label' in self.graph.nodes[key].keys():
                    ci = self.labels.index(self.graph.nodes[key]['label'])
                    fi[p, ci] = 1
                    self.y[(i, p)] = np.zeros(K)
                    self.y[(i, p)][ci] = 1

            self.f[i] = fi

        self.build_relation_matrix()
        self.S = self.build_S()
        # self.labels = labels

    def iterate_f(self):
        # types = list(set(nx.get_node_attributes(self.graph, 'type').values()))
        m = len(self.types)
        # labels = list(set(nx.get_node_attributes(self.graph, 'label').values())) # quais são as classes
        K = len(self.labels)
        f1 = dict()
        alpha = self.alpha
        gamma = self.gamma
        f = self.f
        # types = self.graph.get_types()

        # Iteracao para todos os tipos de objetos
        for i in self.types:
            # nodes_ni = [key for key, attr in self.graph.nodes(data=True) if attr['type'] == i]
            nodes_ni = self.nodes_type[i]
            ni = len(nodes_ni)
            f1[i] = np.zeros((ni, K))
            over = 0

            # Iteracao para cada objeto do tipo i
            for p, key in enumerate(nodes_ni):
                # Se o objeto p do tipo i for rotulado esta restricao estara
                # ativa
                y = self.y.get((i, p), None)
                if y is not None:
                    f1[i][p, :] = alpha * y
                    over += alpha

            # Se houver ligacoes entre os objetos do tipo i com ele
            # mesmo a seguinte parcela contribuira
            Sii = self.S.get((i, i), None)
            if Sii is not None:
                f1[i] += +2 * gamma * np.dot(Sii, f[i])
                over += 2 * gamma

            for j in self.types:
                Sij = self.S.get((i, j), None)
                if Sij is not None:
                    # print('tipo: (%s, %s) - max: %f - min: %f '%(i,j,f1[i].max(),f1[i].min()))
                    f1[i] = f1[i] + gamma * np.dot(Sij, f[j])
                    # print('tipo: (%s, %s) - max: %f - min: %f '%(i,j,f1[i].max(),f1[i].min()))
                    over += gamma

        return f1

    # ---------------------------------------------------------------- #
    # Main method
    # ---------------------------------------------------------------- #
    def run(self, max_it=100):
        t = 0
        # types = set(nx.get_node_attributes(self.graph, 'type').values())
        m = len(self.types)

        # Passo 0
        self.initialize()
        f1 = self.f

        # Passo 1
        while t < max_it:
            if t == 1:
                print('it 2')
            self.f = f1
            f1 = self.iterate_f()
            t += 1

        self.f = f1
        # Passo 3
        c = dict()
        for i in self.types:
            # nodes_ni = [key for key, attr in self.graph.nodes(data=True) if attr['type'] == i]
            nodes_ni = self.nodes_type[i]
            # print(nodes_ni)
            ni = len(nodes_ni)
            c[i] = [-1] * ni

            for p in range(ni):
                c[i][p] = np.argmax(self.f[i][p, :])

        return c

# if __name__ == '__main__':
#     G = read_graph(input_f='graphmodels/graph1600.json')
#     list_doc_nodes = [id for id, t in G.nodes.data('type') if t == 'doc']
#     quant_doc_nodes = len(list_doc_nodes)
#     list_cortes = [int(corte*0.05*quant_doc_nodes) for corte in range(1,10)]
#     quant_partes = int((list_cortes[-1]/2)-1)
#     pre_anotados = []
#     uteis = [id for id, util in G.nodes().data('util') if util == 1]
#     nao_uteis = [id for id, util in G.nodes().data('util') if util == 0]
#     tam_uteis = len(uteis)
#     cortes_uteis = round(tam_uteis/quant_partes)-1
#     locais = [0,1,(tam_uteis-1)] #3
#     locais.extend([i*cortes_uteis for i in range(1,quant_partes-1)]) #12
#     for id in locais:
#         pre_anotados.append(uteis[id])
#         G.nodes[uteis[id]]['label'] = 'util'
#         G.nodes[nao_uteis[id]]['label'] = 'n_util'
#     M = GNetMine(graph = G)
#     c = M.run()
#     print(M.f)
