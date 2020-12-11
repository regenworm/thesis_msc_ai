# models
from n2v import N2VModel
from s2v import S2VModel

# utility
import argparse
import numpy as np
import matplotlib.pyplot as plt
import data_util as du


def add_noise(data, n_remove=5, n_add=5):
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


def train_model(data, model_type, embed_dim, model_fname):
    if model_type == 'n2v':
        nv_emb = N2VModel(embed_dim=embed_dim,
                          emb_name='l2',
                          c_idx=-1,
                          model_fname=model_fname)
        nv_emb.fit(data)
        return nv_emb
    elif model_type == 's2v':
        sv_emb = S2VModel(embed_dim=embed_dim,
                          emb_name='l2',
                          c_idx=-1,
                          model_fname=model_fname)
        sv_emb.fit(data)
        return sv_emb


def run(args):
    # load data
    print(f'== LOADING DATA FROM {args.input} ==')
    data = du.load_edge_list(args.input)
    if args.show_vis:
        v = du.vis_graph(data)
        plt.show(v)

    train_data = add_noise(data)

    # train model
    print(f'== TRAINING {args.model_type} MODEL  ==')
    model = train_model(train_data, args.model_type,
                        args.embed_dim, args.model_fname)

    # classify
    print(f'== TESTING MODEL  ==')
    score = model.score(data)
    print(f'Score for {args.model_type}: {score}')

    return


def parse_args():
    parser = argparse.ArgumentParser(description="Model graph and evaluate.")

    parser.add_argument('--input',
                        default='data/tissue_int.edgelist',
                        help='Input graph path')

    parser.add_argument('--output',
                        default='emb/tissue_model.emb',
                        help='Embeddings path')

    parser.add_argument('--embed_dim',
                        type=int,
                        default=2,
                        help='Number of dimensions. Default is 2.')

    parser.add_argument('--show_vis',
                        dest='show_vis',
                        default=False,
                        action='store_true')

    parser.add_argument('--model_type',
                        type=str,
                        default='n2v',
                        choices=['n2v', 's2v'])

    parser.add_argument('--model_fname',
                        nargs='?',
                        default=None,
                        help='Embeddings path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(args)
