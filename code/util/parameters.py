import argparse


class Parameters:
    input_data = 'data/tissue_int.edgelist'
    train_data_output = 'train_tissue_int.edgelist'
    results_dir = 'results'
    embed_dim = 10
    show_vis = False
    model_type = 'n2v'
    load_model_fname = None
    num_neg_samples_fit = None
    num_neg_samples_score = None
    n_missing_edges = 1
    n_spurious_edges = 1
    directed = False

    # def to_dict(self):
    #     params_dict = {}
    #     params_dict['input_data'] = self.input_data
    #     params_dict['train_data_output'] = self.train_data_output
    #     params_dict['results_dir'] = self.results_dir
    #     params_dict['embed_dim'] = self.embed_dim
    #     params_dict['show_vis'] = self.show_vis
    #     params_dict['model_type'] = self.model_type
    #     params_dict['load_model_fname'] = self.load_model_fname
    #     params_dict['num_neg_samples_fit'] = self.num_neg_samples_fit
    #     params_dict['num_neg_samples_score'] = self.num_neg_samples_score
    #     params_dict['n_missing_edges'] = self.n_missing_edges
    #     params_dict['n_spurious_edges'] = self.n_spurious_edges
    #     return params_dict

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
                        choices=['n2v', 's2v'])

    parser.add_argument('--load_model_fname',
                        nargs='?',
                        default=None,
                        help='Embeddings path')

    parser.add_argument('--num_neg_samples_fit',
                        type=int,
                        default=None,
                        help='Number of negative examples that are sampled to fit the classifier')

    parser.add_argument('--num_neg_samples_score',
                        type=int,
                        default=None,
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
    
    return parser.parse_args()


def get_parameters(notebook=False, **params):
    if notebook:
        return parameters_notebook(params)
    else:
        return parameters_cmdline()
