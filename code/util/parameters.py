import argparse


class Parameters:
    input_data = 'data/tissue_int.edgelist'
    train_data_output = 'train_tissue_int.edgelist'
    results_dir = 'results'
    embed_dim = 10
    show_vis = False
    model_type = 'n2v'
    load_model_fname = None
    num_neg_samples_fit = 500
    num_neg_samples_score = 500
    n_missing_edges = 1
    n_spurious_edges = 1
    directed = False
    generate_data = False
    gen_data_type = 'barabasi'
    gen_data_nodes = 200
    gen_data_edges = 2
    gen_data_alpha = 0.41
    gen_data_beta = 0.54

    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)


def parameters_notebook(params):
    return Parameters(params)


def parameters_cmdline():
    parser = argparse.ArgumentParser(description="Model graph and evaluate.")

    parser.add_argument('--input_data',
                        default='data/tissue_int.edgelist',
                        help='Input graph path')

    parser.add_argument('--train_data_output',
                        default='data/train_tissue_int.edgelist',
                        help='Input graph path')

    parser.add_argument('--results_dir',
                        default='results',
                        help='Results directory')

    parser.add_argument('--embed_dim',
                        type=int,
                        default=10,
                        help='Number of dimensions. Default is 10.')

    parser.add_argument('--show_vis',
                        dest='show_vis',
                        default=False,
                        action='store_true')

    parser.add_argument('--model_type',
                        type=str,
                        default='n2v',
                        choices=['n2v', 's2v', 'netra'])

    parser.add_argument('--load_model_fname',
                        nargs='?',
                        default=None,
                        help='Embeddings path')

    parser.add_argument('--num_neg_samples_fit',
                        type=int,
                        default=500,
                        help='Number of negative examples that are sampled to fit the classifier')

    parser.add_argument('--num_neg_samples_score',
                        type=int,
                        default=500,
                        help='Number of negative examples that are sampled to score the classifier')

    parser.add_argument('--n_missing_edges',
                        type=int,
                        default=1,
                        help='Number of edges removed to create noise')

    parser.add_argument('--n_spurious_edges',
                        type=int,
                        default=1,
                        help='Number of edges added to create noise')

    parser.add_argument('--directed',
                        type=bool,
                        default=False,
                        help='Number of edges added to create noise')

    parser.add_argument('--generate_data',
                        dest='generate_data',
                        default=False,
                        action='store_true')
    
    parser.add_argument('--gen_data_type',
                        type=str,
                        default='n2v',
                        choices=['barabasi', 'scale_free'])

    parser.add_argument('--gen_data_nodes',
                        type=int,
                        default=200,
                        help='When generating data, number of nodes that should be generated. Default is 200.')

    parser.add_argument('--gen_data_edges',
                        type=int,
                        default=2,
                        help='When generating data, number of edges that should be attached to a node. Default is 2.')
    
    parser.add_argument('--gen_data_alpha',
                        type=int,
                        default=0.41,
                        help='Networkx scale free generation alpha parameter. Default is 0.41.')

    parser.add_argument('--gen_data_beta',
                        type=int,
                        default=0.54,
                        help='Networkx scale free generation beta parameter. Default is 0.54.')
    return parser.parse_args()


def get_parameters(notebook=False, **params):
    if notebook:
        return parameters_notebook(params)
    else:
        return parameters_cmdline()
