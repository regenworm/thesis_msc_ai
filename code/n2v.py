# models
from node2vec import Node2Vec
import node2vec.edges as ee

# load model from file
from gensim.models import KeyedVectors

# classifier
from classifier import classify

# utility
import data_util as du


def train_node_embeddings(graph, embedding_dim, fname_model=None):
    node2vec = Node2Vec(graph, dimensions=embedding_dim,
                        walk_length=100, num_walks=50, workers=4)

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
    def __init__(self, embed_dim=2, emb_name='l2', c_idx=0, model_fname=None):
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
        self.nodes = model.wv.vectors

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
        self.nodes = model.wv.vectors

        # edge model
        edge_model = train_edge_embeddings(model, emb_name='l2')
        self.ee_kv = edge_model.as_keyed_vectors()
        self.edges = self.ee_kv.vectors

    def fit(self, data):
        """
        @data: networkx graph
        """
        # if embeddings not loaded
        if not self.from_file:
            self.gen_embeddings(data)

        # fit classifier
        edge_labels = du.construct_embedding_labels(data, self.ee_kv)
        self.clf = classify(self.edges, edge_labels)

    def predict(self, data):
        """
        predict labels for data
        """
        return self.clf.predict(data)

    def score(self, data):
        """
        get score for label prediction of data
        """
        edge_labels = du.construct_embedding_labels(data, self.ee_kv)
        return self.clf.score(self.edges, edge_labels)
