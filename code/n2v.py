# models
from node2vec import Node2Vec
import node2vec.edges as ee

# load model from file
from gensim.models import KeyedVectors

# classifier
from classifier import classify

# utility
import data_util as du
from sklearn.metrics import f1_score
import numpy as np


def train_node_embeddings(graph, embedding_dim, fname_model=None):
    node2vec = Node2Vec(graph, dimensions=embedding_dim,
                        walk_length=10, num_walks=50, workers=4)

    # Embed nodes
    emb_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Save model for later use
    if fname_model is not None:
        emb_model.save(fname_model)
    return emb_model


def train_edge_embeddings(emb_model, emb_name='l2', fname_edge_embs=None):
    if emb_name == 'average':
        edges_embs = ee.AverageEmbedder(keyed_vectors=emb_model.wv)
    elif emb_name == 'hadamard':
        edges_embs = ee.HadamardEmbedder(keyed_vectors=emb_model.wv)
    elif emb_name == 'l1':
        edges_embs = ee.WeightedL1Embedder(keyed_vectors=emb_model.wv)
    else:
        edges_embs = ee.WeightedL2Embedder(keyed_vectors=emb_model.wv)

    # Save embeddings for later use
    if fname_edge_embs is not None:
        edges_embs.as_keyed_vectors().save_word2vec_format(fname_edge_embs)

    return edges_embs


def get_edge_embeddings(data, embed_dim, emb_name='l2'):
    """
    input data, return node features, edge features,
    """
    emb_model = train_node_embeddings(data, embed_dim)
    edge_embeddings = train_edge_embeddings(emb_model, emb_name=emb_name)
    ee_kv = edge_embeddings.as_keyed_vectors()

    return emb_model.wv.vectors, ee_kv.vectors, ee_kv


class N2VModel ():
    def __init__(self, embed_dim=10, emb_name='l2', c_idx=-1, model_fname=None, thresh=0.5):
        """
        @embed_dim: integer, dimensionality of generated embeddings
        @c_idx: integer, determines which classifier from scikit to use
        @emb_name: str, determines which edge embedder from n2v to use
        @model_fname: str, if set, loads w2v format gensim model from file
        """
        self.embed_dim = embed_dim
        self.emb_name = emb_name
        self.classifier_idx = c_idx
        self.from_file = False
        self.thresh = thresh

        if model_fname is not None:
            self.from_file = True
            self.load_model(model_fname)

    def load_model(self, model_fname):
        """
        @model_fname: str, location of gensim word2vec node embeddings model
                      (saved in word2vec format)
        """
        # TODO: load self.emb_name and self.embed_dim
        model = KeyedVectors.load_word2vec_format(model_fname)
        self.nodes = model.wv

        edge_model = train_edge_embeddings(model, emb_name='l2')
        self.ee_kv = edge_model.as_keyed_vectors()
        self.edges = self.ee_kv.vectors
        # still needs classfier from fit

    def gen_embeddings(self, data):
        """
        @data: networkx graph
        """
        # generate embeddings
        # node model
        model = train_node_embeddings(data, self.embed_dim)
        self.nodes = model.wv

        # edge model
        edge_model = train_edge_embeddings(model, emb_name=self.emb_name)
        self.ee_kv = edge_model.as_keyed_vectors()
        self.edges = self.ee_kv.vectors

    def get_embedding(self, edge, keys):
        edge = str(edge)
        n1, n2 = du.edge_str2tuple(edge)
        node1 = self.nodes[n1]
        node2 = self.nodes[n2]
        return np.hstack((node1, node2))

    def get_feature_vectors(self, edges):
        feats = []
        keys = self.ee_kv.vocab.keys()
        # for each edge in data, get feature vector
        for edge in edges:
            feat_vec = self.get_embedding(edge, keys)

            if feat_vec is -1:
                print('Embedding not found')
                continue
            # append to feats
            feats.append(feat_vec)

        return feats

    def fit(self, data):
        """
        @data: networkx graph
        """
        # if embeddings not loaded
        if not self.from_file:
            self.gen_embeddings(data)

        # fit classifier
        # sample balanced classes
        n_data_edges = len(data.edges)
        feats = []
        labels = np.zeros(n_data_edges * 2)
        labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        feats = self.get_feature_vectors(data.edges)
        keys = self.nodes.vocab.keys()
        # negative samples
        for i in range(n_data_edges):
            edge, r_edge = du.sample_edge_idx(data.nodes)

            feat_vec = self.get_embedding(edge, keys)
            if (feat_vec is -1) or (edge in data.edges) or (r_edge in data.edges):
                i -= 1
                continue
            feats.append(feat_vec)

        feats = np.array(feats)
        self.clf = classify(feats, labels)

    def data_to_features(self, data):
        # get all feature vector names (edge1, edge2)
        feats = []
        keys = self.ee_kv.vocab.keys()

        # for each edge in data, get feature vector
        for edge in data.edges:
            feat_vec = None
            # convert edge to string
            edge = str(edge)

            # if edge found save
            feat_vec = self.get_embedding(edge, keys)
            if feat_vec is -1:
                print('Embedding not found')
                continue

            # append to feats
            feats.append(feat_vec)

        feats = np.array(feats)
        return feats

    def predict(self, data):
        """
        predict labels for data
        """
        feats = self.data_to_features(data)
        # gen predictions for data
        preds = self.clf.predict_proba(feats)
        return preds

    def score(self, data):
        """
        get score for label prediction of data
        """
        # sample balanced classes
        n_data_edges = len(data.edges)
        feats = []
        labels = np.zeros(n_data_edges * 2)
        labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        feats = self.get_feature_vectors(data.edges)
        keys = self.ee_kv.vocab.keys()

        # negative samples
        for i in range(n_data_edges):
            edge, r_edge = du.sample_edge_idx(data.nodes)

            feat_vec = self.get_embedding(edge, keys)
            if (feat_vec is -1) or (edge in data.edges) or (r_edge in data.edges):
                i -= 1
                continue
            feats.append(feat_vec)

        feats = np.array(feats)

        predictions = self.clf.predict_proba(feats)
        thresholded = (predictions[:, 1] > self.thresh).astype(int)

        du.plot_prc(self.clf, feats, labels)
        print(predictions)
        f1 = f1_score(labels, thresholded)
        return f1
