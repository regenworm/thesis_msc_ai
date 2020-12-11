import struc2vec as s2v
from gensim.models import KeyedVectors
from n2v import train_edge_embeddings
import data_util as du
from classifier import classify


class S2VModel():
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
        # set arguments
        p = s2v.parse_args()
        args = p.parse_args(['--dimensions', str(self.embed_dim)])
        G = s2v.fnx(data)

        # generate embeddings
        # node model
        s2v.exec_struc2vec(args, G=G)
        model = s2v.learn_embeddings(args)
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



# if __name__ == "__main__":
#     sv = S2VModel()
#     wv = sv.fit('data/tissue_int.edgelist')