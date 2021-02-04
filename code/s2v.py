import models.struc2vec as s2v
from gensim.models import KeyedVectors
from n2v import train_edge_embeddings
from util import data_util as du

import numpy as np
from sklearn.metrics import f1_score
from classifier import classify
import pickle

class S2VModel():
    def __init__(self, embed_dim=2, emb_name='l2', c_idx=-1, thresh=0.5):
        """
        @embed_dim: integer, dimensionality of generated embeddings
        @c_idx: integer, determines which classifier from scikit to use
        @emb_name: str, determines which edge embedder from n2v to use
        @model_fname: str, if set, loads w2v format gensim model from file
        """
        self.model_name = 's2v'
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
        # set arguments
        p = s2v.parse_args()
        args = p.parse_args(['--dimensions', str(self.embed_dim)])
        G = s2v.fnx(data)

        # generate embeddings
        # node model
        s2v.exec_struc2vec(args, G=G)
        model = s2v.learn_embeddings(args)
        self.nodes = model.wv

        # edge model
        edge_model = train_edge_embeddings(model, emb_name='l2')
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

    def score_negative_sampling(self, data, num_samples=None):
        """
        get score for label prediction of data with negative sampling
        """
        n_data_edges = len(data.edges)
        # if not set, sample as many negative edges as there are positive
        if num_samples is None:
            num_samples = n_data_edges

        # gen labels
        all_labels = np.zeros(n_data_edges + num_samples)
        all_labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        true_pos_samples = self.scaler.transform(self.get_feature_vectors(data.edges))


        # get negative samples
        keys = self.ee_kv.vocab.keys()
        neg_edge_names, neg_feats = self.negative_sample(num_samples, data, keys)
        neg_samples = self.scaler.transform(neg_feats)

        # predict
        all_samples = np.vstack((true_pos_samples, neg_samples))
        all_samples_predict = self.clf.predict_proba(all_samples)
        return all_samples_predict, all_labels, neg_edge_names, neg_samples

    def score(self, data):
        """
        get score for label prediction of data
        @fname: output filename for plots
        @dirname: output folder for plots
        """
        samples = self.scaler.transform(self.get_feature_vectors(data))
        preds = self.clf.predict_proba(samples)
        return preds
