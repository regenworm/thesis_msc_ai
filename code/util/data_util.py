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
def save_params(args, fname):
    param_array = []
    input_data = args.input_data
    train_data_output = args.train_data_output
    results_dir = args.results_dir
    embed_dim = args.embed_dim
    show_vis = args.show_vis
    model_type = args.model_type
    load_model_fname = args.load_model_fname
    num_neg_samples_fit = args.num_neg_samples_fit
    num_neg_samples_score = args.num_neg_samples_score
    n_missing_edges = args.n_missing_edges
    n_spurious_edges = args.n_spurious_edges
    directed = args.directed
    
    param_array.append(('input_data', input_data))
    param_array.append(('train_data_output', train_data_output))
    param_array.append(('results_dir', results_dir))
    param_array.append(('embed_dim', embed_dim))
    param_array.append(('show_vis', show_vis))
    param_array.append(('model_type', model_type))
    param_array.append(('load_model_fname', load_model_fname))
    param_array.append(('num_neg_samples_fit', num_neg_samples_fit))
    param_array.append(('num_neg_samples_score', num_neg_samples_score))
    param_array.append(('n_missing_edges', n_missing_edges))
    param_array.append(('n_spurious_edges', n_spurious_edges))
    param_array.append(('directed', directed))

    write_pickle(param_array, fname)


def read_params(fname):
    params_array = read_pickle(fname)
    return dict(params_array)


def load_edge_list(fname, directed):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    return nx.read_edgelist(fname, create_using=G)


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
def generate_data_graph(args):
    """
    returns networkx graph with n nodes
    """
    if args.gen_data_type == 'barabasi':
        graph = nx.barabasi_albert_graph(args.gen_data_nodes,
                                         args.gen_data_edges)
    else:
        graph = nx.scale_free_graph(args.gen_data_nodes,
                                    args.gen_data_alpha,
                                    args.gen_data_beta)
        args.directed = True
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
        tsne = TSNE(random_state=0, n_iter=5000, metric='cosine')
        data2d = tsne.fit_transform(data)
    return data2d


def add_noise(data, n_missing_edges=50, n_spurious_edges=50):
    """
    @data: networkx graph
    """
    new_data = deepcopy(data)
    print('???')

    # missing edges
    if nx.is_directed(new_data):
        # convert from [node, node, direction] format to [node1, node2] format
        edges = np.array([(edge[0], edge[1]) if edge[2] == 0 else (edge[1], edge[0]) for edge in data.edges])
    else:
        edges = np.array([(edge[0], edge[1]) for edge in data.edges])

    # randomly select edges to remove
    r_idx = np.random.choice(range(len(edges)-1), n_missing_edges)
    # create edge tuples
    missing_edges = np.array(edge_list2edge_tuple(edges[r_idx]))
    # remove from graph
    new_data.remove_edges_from(missing_edges)

    # spurious edges
    nodes = np.array(data.nodes)
    # randomly select nodes to connect
    a_idx = np.random.choice(range(len(nodes)-1), n_spurious_edges * 2)
    # iterate over selected nodes in pairs
    it = iter(nodes[a_idx])
    spurious_edges = np.array([(edge, next(it)) for edge in it])
    # add edges to graph
    for i in range(0, len(a_idx), 2):
        new_data.add_edges_from(spurious_edges)

    # ensure connected
    if not nx.is_directed(new_data):
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
    # unpacked_edge_names = []
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
        # unpacked_edge_names.append(run_edge_names)
    return np.array(unpacked_preds), run_edge_names, unpacked_labels


def unpack_neg_samples(all_runs_predicions):
    unpacked_edge_names = []
    unpacked_embeddings = []
    for bootstrap in all_runs_predicions:
        run_embs = []
        run_edge_names = []

        for edge_name, emb in bootstrap:
            run_edge_names.append(edge_name)
            run_embs.append(emb)

        unpacked_edge_names.append(run_edge_names)
        unpacked_embeddings.append(run_embs)
    return unpacked_edge_names, unpacked_embeddings


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
    return (str(n[0]), str(n[1])),  (str(n[1]), str(n[0]))

def check_directed(graph):
    return nx.is_directed(graph)
