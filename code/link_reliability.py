from utility import read_data, to_numpy

G = read_data()
am = to_numpy(G)

for link_label in range(am.shape[0]):
    am[link_label][link_label]

    break
