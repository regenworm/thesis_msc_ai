import matplotlib.pyplot as plt
import networkx as nx
import os


def read_data(fpath=[]):
    # set default path to file
    if len(fpath) == 0:
        dirname = os.path.dirname(__file__)
        fpath = [dirname, 'data', 'karate', 'karate.gml']
    filename = os.path.join(*fpath)

    # read file
    G = nx.read_gml(filename, label='id')
    return G

def to_numpy(graph):
    return nx.to_numpy_matrix(graph)

def draw_graph(graph, graph_name, fpath=[], options={}):
    # set default path to file
    if len(fpath) == 0:
        dirname = os.path.dirname(__file__)
        fpath = [dirname, 'data', 'figures', '%s.png' % graph_name]
    filename = os.path.join(*fpath)

    # draw in matplotlib and save to location
    nx.draw(graph, with_labels=True, font_weight='bold', **options)
    plt.savefig(filename)

    print('Saved file at %s' % filename)


# G = read_data()
# draw_graph(G, 'karate_test')
