import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import gensim as gs
from sklearn.manifold import TSNE
import numpy as np
import re
from sklearn.metrics import plot_precision_recall_curve

# some theme compatability issues
pal = sns.color_palette('Set2')
cmap = sns.color_palette('Set2', as_cmap=True)
c1 = cmap(0)
c2 = cmap(1)
sns.set_theme(palette=pal)
graph_vis_options = {
    "font_size": 24,
    "node_size": 1000,
    "linewidths": 20,
    "arrows": True,
    "width": 5,
    "node_color": c1,
    "edge_color": c2
}


def simulate_data(n, m):
    """
    returns networkx graph with n nodes
    """
    graph = nx.barabasi_albert_graph(n, m)
    return graph


def load_edge_list(fname='data/tissue_int.edgelist'):
    return nx.read_edgelist(fname)


def read_embedding(fname):
    return gs.models.KeyedVectors.load_word2vec_format(fname)


def vis_graph(graph, set_labels=True, options=graph_vis_options):
    """
    visualise graph through networkx draw
    """
    position = nx.spring_layout(graph)
    if set_labels:
        labels = {node: node for node in graph.nodes()}
    else:
        labels = {node: '' for node in graph.nodes()}
    nodes = nx.draw(graph, position, labels=labels)
    return nodes


def data_to_2d(data):
    """
    Project input data to 2D space with tSNE
    """
    if data.shape[1] == 2:
        data2d = data
    else:
        tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
        data2d = tsne.fit_transform(data)
    return data2d


def vis_embeddings(data, options={}):
    """
    Project embeddings in 2D space and return visualisation
    """
    # project to 2d space
    embeddings2d = data_to_2d(data)

    # plot
    fig = plt.figure()
    plt.scatter(embeddings2d[:, 0], embeddings2d[:, 1])
    return fig


def vis_labeled_embeddings(data, labels, color_list, options={}):
    """
    Project embeddings in 2D space and return visualisation with a legend

    Edge embeddings exist for all possible edges,
    class 1 exists in the data and class 0 does not
    """
    embeddings2d = data_to_2d(data)

    fig = plt.figure()
    # for each unique label, add to datapoint
    for label in np.unique(labels):
        label_indices = np.where(label == labels)

        plt.scatter(embeddings2d[label_indices, 0],
                    embeddings2d[label_indices, 1],
                    c=color_list[label_indices], label=label)
    plt.legend()
    return fig


def edge_str2tuple(idx):
    tuple_idx = re.sub('[^a-zA-Z\d\s:]', '', idx).split(' ')
    return tuple_idx[0], tuple_idx[1]


def construct_embedding_labels(graph, edge_embedding_kv):
    labels = []
    for edge_idx in edge_embedding_kv.vocab:
        u, v = edge_str2tuple(edge_idx)
        if graph.has_edge(u, v):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)


def vis_edge_embeddings(data, edges, edge_labels):
    """
    @data: networkx graph
    @ee_kv: edge embeddings as keyed vector, trained on data input

    returns figure of 2d projected edge embeddings
    """
    label_colors = cmap(edge_labels)
    fig = vis_labeled_embeddings(edges, edge_labels, label_colors)
    return fig


def sample_edge_idx(nodes):
    n = np.random.choice(range(len(nodes)), 2)
    return f"('{n[0]}', '{n[1]}')", f"('{n[1]}', '{n[0]}')"

def add_noise(data, n_remove=50, n_add=50):
    """
    @data: networkx graph
    """
    new_data = data.copy()

    # missing edges
    edges = np.array(data.edges)
    r_idx = np.random.choice(range(len(edges)), n_remove)
    new_data.remove_edges_from(edges[r_idx])

    # spurious edges
    nodes = np.array(data.nodes)
    a_idx = np.random.choice(range(len(nodes)), n_add * 2)
    s_edge = nodes[a_idx]

    for i in range(0, len(a_idx), 2):
        new_data.add_edge(s_edge[i], s_edge[i+1])

    return new_data

def plot_prc(clf, xt, yt):
    prc_curve = plot_precision_recall_curve(clf, xt, yt)
    plt.savefig('plots/prc_emb.png')
    return prc_curve
