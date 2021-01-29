import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from os import path
from . import data_util as du
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix


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


def vis_embeddings(data, options={}):
    """
    Project embeddings in 2D space and return visualisation
    """
    # project to 2d space
    embeddings2d = du.data_to_2d(data)

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
    embeddings2d = du.data_to_2d(data)

    fig = plt.figure()
    # for each unique label, add to datapoint
    for label in np.unique(labels):
        label_indices = np.where(label == labels)

        plt.scatter(embeddings2d[label_indices, 0],
                    embeddings2d[label_indices, 1],
                    c=color_list[label_indices], label=label)
    plt.legend()
    return fig


def vis_edge_embeddings(data, edges, edge_labels):
    """
    @data: networkx graph
    @ee_kv: edge embeddings as keyed vector, trained on data input

    returns figure of 2d projected edge embeddings
    """
    label_colors = cmap(edge_labels)
    fig = vis_labeled_embeddings(edges, edge_labels, label_colors)
    return fig


def plot_prc(clf, xt, yt, fname, dirname):
    prc_curve = plot_precision_recall_curve(clf, xt, yt)
    plt.savefig(path.join(dirname, 'prc_' + fname))

    return prc_curve


def plot_cf(clf, xt, yt, fname, dirname):
    cf_mat = plot_confusion_matrix(clf, xt, yt)
    plt.savefig(path.join(dirname, 'cf_' + fname))

    return cf_mat


def plot_heatmap(data, fname, xticklabels='auto', dirname):
    plt.figure(figsize=(11, 7))
    plt.clf()
    ax = sns.heatmap(data)
    ax.set_xticklabels(xticklabels, rotation=30)
    plt.xlabel = 'Edge label'
    plt.ylabel = 'Run number'
    plt.savefig(path.join(dirname, 'hm_' + fname))
    plt.close()
    return ''


def plot_metrics(clf, xt, yt, fname, dirname):
    prc_curve = plot_prc(clf, xt, yt, fname,  dirname)
    plt.close(prc_curve.figure_)

    cf_mat = plot_cf(clf, xt, yt, fname,  dirname)
    plt.close(cf_mat.figure_)
