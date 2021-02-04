# models
from node2vec import Node2Vec
import node2vec.edges as ee

# load model from file
from gensim.models import KeyedVectors

# classifier
from classifier import classify
import pickle

# utility
from util import data_util as du
from sklearn.metrics import f1_score
import numpy as np


def train_node_embeddings(graph, embedding_dim, fname_model=None):
    node2vec = Node2Vec(graph, dimensions=embedding_dim,
                        walk_length=10, num_walks=20, workers=4)

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


class N2VModel ():
    def __init__(self, embed_dim=10, emb_name='l2', c_idx=-1, thresh=0.5):
        """
        @embed_dim: integer, dimensionality of generated embeddings
        @c_idx: integer, determines which classifier from scikit to use
        @emb_name: str, determines which edge embedder from n2v to use
        @model_fname: str, if set, loads w2v format gensim model from file
        """
        self.model_name = 'n2v'
        self.embed_dim = embed_dim
        self.emb_name = emb_name
        self.classifier_idx = c_idx
        self.from_file = False
        self.thresh = thresh

    @staticmethod
    def load_model(fname):
        with open(fname, 'rb') as f:
            model = pickle.load(f)
            return model

    def save_model(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

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
        feats = np.array(feats)
        return feats

    def negative_sample(self, n_edges, data, keys):
        feats = []
        edge_names = []
        # negative samples
        for i in range(n_edges):
            edge, r_edge = du.sample_edge_idx(data.nodes)

            feat_vec = self.get_embedding(edge, keys)
            if (feat_vec is -1) or (edge in data.edges) or (r_edge in data.edges):
                i -= 1
                continue
            feats.append(feat_vec)
            edge_names.append(edge)

        feats = np.array(feats)
        return edge_names, feats

    def fit(self, data, num_samples=None):
        """
        @data: networkx graph
        """
        # fit classifier
        # sample balanced classes
        n_data_edges = len(data.edges)
        if num_samples is None:
            num_samples = n_data_edges
        feats = []
        labels = np.zeros(n_data_edges + num_samples)
        labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        feats = self.get_feature_vectors(data.edges)
        keys = self.nodes.vocab.keys()
        neg_edge_names, neg_feats  = self.negative_sample(num_samples, data, keys)
        feats = np.vstack((feats, neg_feats))
        self.clf, self.scaler = classify(feats, labels)

        return neg_edge_names, neg_feats

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

    def score_negative_sampling(self, data, neg_edge_names):
        """
        get score for label prediction of data with negative sampling
        """
        n_data_edges = len(data.edges)
        n_neg_samples = len(neg_edge_names)

        # gen labels
        all_labels = np.zeros(n_data_edges + n_neg_samples)
        all_labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        true_pos_samples = self.scaler.transform(self.get_feature_vectors(data.edges))
        neg_samples = self.scaler.transform(self.get_feature_vectors(neg_edge_names))

        # predict
        all_samples = np.vstack((true_pos_samples, neg_samples))
        all_samples_predict = self.clf.predict_proba(all_samples)
        return all_samples_predict, all_labels

    def score(self, data):
        """
        get score for label prediction of data
        @fname: output filename for plots
        @dirname: output folder for plots
        """
        samples = self.scaler.transform(self.get_feature_vectors(data))
        preds = self.clf.predict_proba(samples)
        return preds
