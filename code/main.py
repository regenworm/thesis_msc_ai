from n2v import train_node_embeddings, train_edge_embeddings
from classifier import classify
import data_util as du
import matplotlib.pyplot as plt


def gen_dataset(N, embed_dim, m=2):
    new_data = du.simulate_data(N, m)

    new_emb_model = train_node_embeddings(new_data, embed_dim)
    new_edge_embeddings = train_edge_embeddings(new_data, new_emb_model)

    new_ee_kv = new_edge_embeddings.as_keyed_vectors()
    labels = du.construct_embedding_labels(new_data, new_ee_kv)
    tdata = {'data': new_data, 'features': new_ee_kv.vectors, 'labels': labels}
    return tdata


def get_edge_embeddings(data, embed_dim, emb_name='l2'):
    """
    input data, return node features, edge features,
    """
    emb_model = train_node_embeddings(data, embed_dim)
    edge_embeddings = train_edge_embeddings(data, emb_model, emb_name=emb_name)
    ee_kv = edge_embeddings.as_keyed_vectors()

    return emb_model.wv.vectors, ee_kv.vectors, ee_kv


if __name__ == "__main__":
    num_nodes = 50
    embed_dim = 10
    show_vis = False

    # simdata pipeline
    # # Load data
    # data = du.simulate_data(num_nodes, 2)
    # num_edges = len(data.edges)
    # if show_vis:
    #     v = du.vis_graph(data)
    #     plt.show(v)
    # print(f'The data contains {num_nodes} nodes, and {num_edges} edges')

    # # train model
    # nodes, edges, ee_kv = get_edge_embeddings(data, embed_dim)
    # edge_labels = du.construct_embedding_labels(data, ee_kv)
    # print(f'The network node data is now embedded into {embed_dim} dimensions')

    # # visualise features
    # if show_vis:
    #     fignodes = du.vis_embeddings(nodes)
    #     plt.show(fignodes)
    #     figedges = du.vis_edge_embeddings(data, edges, edge_labels)
    #     plt.show(figedges)

    # # classify features
    # clf = classify(edges, edge_labels)
    # print(clf.score(edges, edge_labels))

