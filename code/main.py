import data_util as du
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder


def train_node_embeddings(graph, fname_model=None):
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=10, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Save model for later use
    if not fname_model is None:
        model.save(fname_model)
    return model

# def node_embeddings(graph, model, fname_node_embs=None):
#     # # Look for most similar nodes
#     # model.wv.most_similar('2')  # Output node names are always strings

#     # Save embeddings for later use
#     if not fname_node_embs is None:
#         model.wv.save_word2vec_format(fname_node_embs)

#     return model.wv


def train_edge_embeddings(graph, model, fname_edge_embs=None):# Embed edges using Hadamard method
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    edges_kv = edges_embs.as_keyed_vectors()

    # Save embeddings for later use
    if not fname_edge_embs is None:
        edges_kv.save_word2vec_format(fname_edge_embs)

    return edges_kv

if __name__ == "__main__":
    EMBEDDING_FILENAME = 'sim.emb'
    EMBEDDING_MODEL_FILENAME = 'sim_model.mdl'
    graph = du.simulate_data(100, 50)
    du.vis_graph(graph)

    model = train_node_embeddings(graph)
    edge_embeddings = train_edge_embeddings(graph, model)

    du.vis_embeddings(edge_embeddings)

