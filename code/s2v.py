import models.struc2vec as s2v
from gensim.models import KeyedVectors
from n2v import train_edge_embeddings
import data_util as du
import numpy as np
from sklearn.metrics import f1_score
from classifier import classify


class S2VModel():
    def __init__(self, embed_dim=2, emb_name='l2', c_idx=-1, model_fname=None, thresh=0.5):
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

            if feat_vec == -1:
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
            if (feat_vec == -1) or (edge in data.edges) or (r_edge in data.edges):
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
            if feat_vec == -1:
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
            if (feat_vec == -1) or (edge in data.edges) or (r_edge in data.edges):
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
