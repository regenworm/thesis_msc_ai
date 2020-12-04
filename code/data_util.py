# graph utilities
import networkx as nx
import pandas as pd

# visualise graph
import matplotlib.pyplot as plt
import seaborn as sns

# visualise embeddings with t-SNE
from sklearn.manifold import TSNE
 
# Create a graph
def simulate_data(n,m):
    graph = nx.barabasi_albert_graph(n, m)
    return graph

def vis_graph(graph):
    sns.set_theme()
    edges = graph.edges
    nx.draw(graph)
    plt.show()

def vis_embeddings(embeddings):
    tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
    embeddings2d = tsne.fit_transform(embeddings)

    plt.scatter(embeddings2d[:,0], embeddings2d[:,1])
    plt.show()
    return embeddings2d