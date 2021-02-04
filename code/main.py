# models
from n2v import N2VModel
from s2v import S2VModel

# utility
import numpy as np

# import data_util as du
from util import data_util
from util import file_util
from util import parameters
from os import path


def train_model(data, model_type, embed_dim):
    if model_type == 'n2v':
        nv_emb = N2VModel(embed_dim=embed_dim,
                          emb_name='l2',
                          c_idx=-1)
        nv_emb.gen_embeddings(data)
        return nv_emb
    elif model_type == 's2v':
        sv_emb = S2VModel(embed_dim=embed_dim,
                          emb_name='l2',
                          c_idx=-1)
        sv_emb.gen_embeddings(data)
        return sv_emb


def run(args):
    # create run results directory and save metadata
    run_dir = file_util.create_run_dir(results_dir=args.results_dir)
    args.run_dir = run_dir
    data_util.save_params(args, path.join(run_dir, 'data', 'params_run.bin'))

    # load data
    print(f'== LOADING DATA FROM {args.input_data} ==')
    data = data_util.load_edge_list(args.input_data)

    # add noise, and save modified data
    n_missing_edges = args.n_missing_edges
    n_spurious_edges = args.n_spurious_edges
    train_data, missing, spurious = data_util.add_noise(data, n_missing_edges, n_spurious_edges)
    data_util.write_edge_list(train_data, path.join(run_dir, 'data', 'train_data.edgelist'))
    data_util.write_pickle(missing.tolist(), path.join(run_dir, 'data', 'missing.bin'))
    data_util.write_pickle(spurious.tolist(), path.join(run_dir, 'data', 'spurious.bin'))

    # train model
    # generates embeddings
    print(f'== TRAINING {args.model_type} MODEL  ==')
    model = train_model(train_data, args.model_type,
                        args.embed_dim)

    # classify
    # fit classifier and test model
    print(f'== TESTING MODEL  ==')

    # for each bootstrap run
    # collect predictions for missing and spurious links
    bootstrap_preds_missing = []
    bootstrap_preds_spurious = []
    bootstrap_preds_all = []

    # for each bootstrap run, for all negative samples
    # collect list of tuples with (index of embedding, embedding)
    fit_negative_samples = []
    score_negative_samples = []

    # test each bootstrap run on the same negative samples
    # generate negative samples and save these in an array
    keys = model.ee_kv.vocab.keys()
    score_neg_edge_names, score_neg_feats = model.negative_sample(args.num_neg_samples_clf, data, keys)
    score_negative_samples.append(
        [(idx, emb) for idx, emb in zip(score_neg_edge_names, score_neg_feats)]
    )

    print("== STARTING BOOTSTRAP ==")
    # do bootstrappping
    for i in range(20):
        # fit and save negative samples
        fit_run_neg_edge_names, fit_run_neg_samples = model.fit(train_data)
        fit_negative_samples.append(
            [(idx, emb) for idx, emb in zip(fit_run_neg_edge_names, fit_run_neg_samples)]
        )

        # score edges and save predictions and negative samples
        all_preds, all_labels = model.score_negative_sampling(data, score_neg_edge_names)
        edge_names = list(data.edges) + score_neg_edge_names
        bootstrap_preds_all.append(
            [(edge_name, preds, label) for edge_name, preds, label in zip(edge_names, all_preds, all_labels)]
        )

        # score missing and spurious
        if int(args.n_missing_edges) > 0:
            preds_missing = model.score(missing)
            bootstrap_preds_missing.append(
                [(edge_name, preds, label) for edge_name, preds, label in zip(missing, preds_missing, np.ones(len(missing)))]
            )

        if int(args.n_spurious_edges) > 0:
            preds_spurious = model.score(spurious)
            bootstrap_preds_spurious.append(
                [(edge_name, preds, label) for edge_name, preds, label in zip(spurious, preds_spurious, np.zeros(len(spurious)))]
            )

    data_util.write_pickle(bootstrap_preds_missing, path.join(run_dir, 'data', 'bootstrap_preds_missing.bin'))
    data_util.write_pickle(bootstrap_preds_spurious, path.join(run_dir, 'data', 'bootstrap_preds_spurious.bin'))
    data_util.write_pickle(bootstrap_preds_all, path.join(run_dir, 'data', 'bootstrap_preds_all.bin'))
    data_util.write_pickle(fit_negative_samples, path.join(run_dir, 'data', 'fit_negative_samples.bin'))
    data_util.write_pickle(score_negative_samples, path.join(run_dir, 'data', 'score_negative_samples.bin'))

    # save model binary
    model.save_model(path.join(run_dir, 'model_output', 'model.bin'))
    print(f"== FINISHED, SAVED RUN: {run_dir} ==")
    return

if __name__ == "__main__":
    args = parameters.get_parameters()

    run(args)
