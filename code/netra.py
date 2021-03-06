# models
from models.netra.src import train
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


def train_node_embeddings(data, seed, outputdir, emb_dim):
    embeddings = train.train_netra(data, seed, outputdir, emb_dim)
    print(embeddings)
    return embeddings


class NetRAModel ():
    def __init__(self, embed_dim=10, c_idx=-1, thresh=0.5, seed=0):
        """
        @embed_dim: integer, dimensionality of generated embeddings
        @c_idx: integer, determines which classifier from scikit to use
        @emb_name: str, determines which edge embedder from n2v to use
        @model_fname: str, if set, loads w2v format gensim model from file
        """
        self.model_name = 'netra'
        self.embed_dim = embed_dim
        self.classifier_idx = c_idx
        self.from_file = False
        self.thresh = thresh
        self.seed = seed

    @staticmethod
    def load_model(fname):
        with open(fname, 'rb') as f:
            model = pickle.load(f)
            return model

    def save_model(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def gen_embeddings(self, data, outputdir):
        """
        @data: networkx graph
        """
        # generate embeddings
        # node model
        embeddings = train_node_embeddings(data, self.seed, outputdir, self.embed_dim)
        self.nodes = embeddings

        # # edge model
        # edge_model = train_edge_embeddings(model, emb_name=self.emb_name)
        # self.ee_kv = edge_model.as_keyed_vectors()
        # self.edges = self.ee_kv.vectors

    def get_embedding(self, edge):
        edge = str(edge)
        n1, n2 = du.edge_str2tuple(edge)
        node1 = self.nodes[int(n1)]
        node2 = self.nodes[int(n2)]
        return np.hstack((node1, node2))

    def get_feature_vectors(self, edges):
        feats = []
        # for each edge in data, get feature vector
        for edge in edges:
            feat_vec = self.get_embedding(edge)

            if feat_vec is -1:
                print('Embedding not found')
                continue
            # append to feats
            feats.append(feat_vec)
        feats = np.array(feats)
        return feats

    def negative_sample(self, n_edges, data):
        feats = []
        edge_names = []
        # negative samples
        iterations = list(range(n_edges))
        for i in iterations:
            edge, r_edge = du.sample_edge_idx(data.nodes)

            feat_vec = self.get_embedding(edge)
            if du.check_directed(data):
                if (feat_vec is -1) or (edge in data.edges):
                    iterations.append(n_edges)
                    continue
            else:
                if (feat_vec is -1) or (edge in data.edges) or (r_edge in data.edges):
                    iterations.append(n_edges)
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
        neg_edge_names, neg_feats  = self.negative_sample(num_samples, data)
        feats = np.vstack((feats, neg_feats))
        self.clf, self.scaler = classify(feats, labels)

        return neg_edge_names, neg_feats

    def data_to_features(self, data):
        # get all feature vector names (edge1, edge2)
        feats = []

        # for each edge in data, get feature vector
        for edge in data.edges:
            feat_vec = None
            # convert edge to string
            edge = str(edge)

            # if edge found save
            feat_vec = self.get_embedding(edge)
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
        tuple_data = du.edge_list2edge_tuple(data)
        samples = self.scaler.transform(self.get_feature_vectors(tuple_data))
        preds = self.clf.predict_proba(samples)
        return preds
