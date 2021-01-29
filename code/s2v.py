import models.struc2vec as s2v
from gensim.models import KeyedVectors
from n2v import train_edge_embeddings
import data_util as du
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
        indices = []
        # negative samples
        for i in range(n_edges):
            edge, r_edge = du.sample_edge_idx(data.nodes)

            feat_vec = self.get_embedding(edge, keys)
            if (feat_vec is -1) or (edge in data.edges) or (r_edge in data.edges):
                i -= 1
                continue
            feats.append(feat_vec)
            indices.append(edge)

        feats = np.array(feats)
        return feats, indices

    def fit(self, data):
        """
        @data: networkx graph
        """
        # fit classifier
        # sample balanced classes
        n_data_edges = len(data.edges)
        feats = []
        labels = np.zeros(n_data_edges * 2)
        labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        feats = self.get_feature_vectors(data.edges)
        keys = self.nodes.vocab.keys()
        neg_feats, indices = self.negative_sample(n_data_edges, data, keys)
        feats = np.vstack((feats, neg_feats))
        self.clf, self.scaler = classify(feats, labels)

        return neg_feats, indices

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

    def score(self, data, missing, spurious, it=''):
        """
        get score for label prediction of data
        """
        it = str(it)
        # sample balanced classes
        n_data_edges = len(data.edges)
        labels = np.zeros(n_data_edges * 2)
        labels[:n_data_edges] = 1

        # for each edge in data, get feature vector
        true_pos_samples = self.scaler.transform(self.get_feature_vectors(data.edges))


        # negative samples
        keys = self.ee_kv.vocab.keys()
        neg_feats, indices = self.negative_sample(n_data_edges, data, keys)
        neg_samples = self.scaler.transform(neg_feats)

        # # + samples
        # du.plot_metrics(self.clf, true_pos_samples, labels[:n_data_edges], fname='emb_pos'+ it +'.png')
        # # - samples
        # du.plot_metrics(self.clf, np.array(neg_samples), labels[n_data_edges:], fname='emb_neg'+it+'.png')

        # all samples
        du.plot_metrics(self.clf, np.vstack((true_pos_samples, neg_samples)), labels, fname=f'{self.model_name}_emb_all'+it+'.png')

        # missing, spurious
        # get missing/spurious features
        n_noisy_samples = len(missing)
        noisy_samples = self.scaler.transform(self.get_feature_vectors(missing + spurious))
        noisy_sample_labels = np.zeros(n_noisy_samples * 2)
        noisy_sample_labels[:n_noisy_samples] = 1
        du.plot_metrics(self.clf, noisy_samples, noisy_sample_labels, fname=f'{self.model_name}_emb_noisy'+it+'.png')

        preds_missing = self.clf.predict_proba(noisy_samples[:n_noisy_samples])
        preds_spurious = self.clf.predict_proba(noisy_samples[n_noisy_samples:])
        return preds_missing, preds_spurious, neg_feats, indices
