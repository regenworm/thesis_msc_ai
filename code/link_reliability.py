from utility import read_data, to_numpy
import numpy as np

def get_group(node, partitions):
    return partitions[node]


def get_observed_links(group_alpha, group_beta, observed_network):
    observed_links = 0
    for node_alpha in group_alpha:
        for node_beta in group_beta:
            if observed_network[node_alpha][node_beta] > 0:
                observed_network += 1
            # if observed_network[node_beta][node_alpha] > 0:
            #     observed_network += 1
    return observed_links


def get_max_links(group_alpha, group_beta):
    max_links = len(group_alpha) * len(group_beta)

    return max_links



def hyp(partitions, max_links, observed_links):
    binomial = max_links, observed_links
    hyp = np.log(max_links + 1) + np.log(binomial)
    return np.sum(hyp)


def get_normalizing_constant(partitions, max_links, observed_links):
    return np.sums(np.exp(- hyp(partitions, max_links, observed_links)))



def link_reliability(node_i, node_j, observed_network):
    partitions = []
    group_alpha = get_group(node_i, partitions)
    group_beta = get_group(node_j, partitions)

    observed_links_ij = get_observed_links(group_alpha, group_beta, observed_network)
    max_links_ij = get_max_links(group_alpha, group_beta)
    H_P = hyp(partitions)
    Z = 1/(get_normalizing_constant(partitions, max_links_ij, observed_links_ij))

   return Z* np.sum(((observed_links_ij + 1)/(max_links_ij + 2))*np.exp(-H_P))

def sample_partitions(data, n_of_nodes):
    

# G = read_data()
# am = to_numpy(G)

# for link_label in range(am.shape[0]):
#     am[link_label][link_label]

#     break
