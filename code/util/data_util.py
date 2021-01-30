# Contains utility functions for network data
# e.g. simulating, loading, converting, generating labels, projecting,
# adding noise etc.
import networkx as nx
import gensim as gs
from sklearn.manifold import TSNE
import numpy as np
import pickle
import re
from copy import deepcopy

# IO functions
def load_edge_list(fname):
    return nx.read_edgelist(fname)


def write_edge_list(graph, fname):
    return nx.write_edgelist(graph, fname)


def read_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def write_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def read_embedding(fname):
    return gs.models.KeyedVectors.load_word2vec_format(fname)


# data generation functions
def simulate_data(n, m):
    """
    returns networkx graph with n nodes
    """
    graph = nx.barabasi_albert_graph(n, m)
    return graph


def construct_embedding_labels(graph, edge_embedding_kv):
    labels = []
    for edge_idx in edge_embedding_kv.vocab:
        u, v = edge_str2tuple(edge_idx)
        if graph.has_edge(u, v):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)


# data modification functions
def data_to_2d(data):
    """
    Project input data to 2D space with tSNE
    """
    if data.shape[1] == 2:
        data2d = data
    else:
        tsne = TSNE(random_state=0, n_iter=000, metric='cosine')
        data2d = tsne.fit_transform(data)
    return data2d


def add_noise(data, n_remove=50, n_add=50):
    """
    @data: networkx graph
    """
    new_data = deepcopy(data)

    # missing edges
    edges = np.array(data.edges)
    r_idx = np.random.choice(range(len(edges)-1), n_remove)
    missing_edges = np.array(edge_list2edge_tuple(edges[r_idx]))
    new_data.remove_edges_from(missing_edges)

    # spurious edges
    nodes = np.array(data.nodes)
    a_idx = np.random.choice(range(len(nodes)-1), n_add * 2)
    it = iter(nodes[a_idx])
    spurious_edges = np.array([(edge, next(it)) for edge in it])
    for i in range(0, len(a_idx), 2):
        new_data.add_edges_from(spurious_edges)

    # ensure connected
    edges = list(nx.k_edge_augmentation(new_data, 1))
    new_data.add_edges_from(edges)

    return new_data, missing_edges, spurious_edges


# data format conversion functions
def edge_str2tuple(idx):
    tuple_idx = re.sub('[^a-zA-Z\d\s:]', '', idx).split(' ')
    return tuple_idx[0], tuple_idx[1]


def tuple2edge_str(tuples):
    edge_str = [f"('{t[0]}', '{t[1]}')" for t in tuples]
    return edge_str


def edge_list2edge_tuple(edges):
    return [(e1, e2) for e1, e2 in edges]


def unpack_values(all_runs_predicions):
    unpacked_preds = []
    unpacked_labels = []
    for bootstrap in all_runs_predicions:
        run_preds = []
        run_labels = []
        run_edge_names = []

        for edge_name, preds, label in bootstrap:
            run_preds.append(preds)
            run_labels.append(label)
            run_edge_names.append(edge_name)

        unpacked_preds.append(run_preds)
        unpacked_labels.append(run_labels)
    return np.array(unpacked_preds), run_edge_names, unpacked_labels


def filter_preds_by_label(all_runs_preds, labels):
    labels = np.array(labels).astype(int)
    label_filtered = []
    for run_idx, run_labels in enumerate(labels):
        label_filtered_run = []
        for node_idx, node_labels in enumerate(run_labels):
            label_filtered_run.append(all_runs_preds[run_idx, node_idx, node_labels])
        label_filtered.append(label_filtered_run)
    all_runs_preds = np.array(label_filtered)
    return all_runs_preds


# data utility functions
def sample_edge_idx(nodes):
    n = np.random.choice(range(len(nodes)-1), 2)
    return f"('{n[0]}', '{n[1]}')", f"('{n[1]}', '{n[0]}')"