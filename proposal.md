# Method Extension
## ORIGINAL IDEA: Feature engineering
Train embeddings and enhance performance of the model by adding features that are descriptive for biological datasets specifically. Possible features so far (taken from Barabasi table):
- Average degree
- Number of nodes
- Average path length
- Clustering coefficient

### TODO
- Literature research showing what features have already been done

## IDEA 2: Regularized VAE
Based on section VI B of comprehensive graph survey.  GraphVAE, Regularized GraphVAE, MolGAN, NetRA show a similar idea.

*Network data -> Encoder with final softmax layer -> outputs vector in latent space -> decoder sequentially generate nodes and edges to construct local graph*

*The encoder has a final softmax layer such that it can learn a classification which should hold across datasets.*

*Regularize latent space adversarially (NetRA shows how this works)*

GraphVAE takes in a network, node features and edge features. It can generate a graph together with all these features. Regularized GraphVAE allows you to add more constraints on certain features of these graphs. However the latent space where the data is projected is not meaningful. It is also limited in scale.
https://arxiv.org/abs/1802.03480

Figure 1 of the MolGAN paper very concisely shows the idea of MolGAN. However, this is completely GAN based with a Reinforcement Learning reward (probably for better backpropagation?). This means that it only generates graphs, in this case chemical compounds. This makes it hard to infer edge scores from this type of system.
https://arxiv.org/pdf/1805.11973.pdf

NetRA is very similar, figure 1 of their paper shows their architecture clearly (section 3.2 of their paper explains specifically about the regularization with the GAN structure). A GAN is used to regularize the encoder output so it is forced to learn useful information about the data.
https://sites.cs.ucsb.edu/~bzong/doc/kdd-18.pdf


### TODO
- Is this architecture viable (latent dimension softmax vector -> decoder -> local graph), as it is similar to random graphs (node class determines connectivity). Possibly look at GraphVAE and MolGAN more closely, generate adjacency matrix, edge attributes, and node features. This would make using multiple datasets harder.
- How many classes? hyperparameter -> start at 2
- One problem that i will run into is that different datasets have different powerlaw distributions (different tao). How to compensate for this? I've found a few papers, not sure which are relevant:
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7188358/
    - https://www.tandfonline.com/doi/full/10.1080/13658810412331280130?casa_token=UCafGuTEhAEAAAAA%3AzXK7gYfx5W3AMhwhgWAEH15Vk0mfxXZCdGRa0ZPS8jOA1q9Y7-koahLDmK7iw3evIEFg4Obq6dkrYg
    - https://ieeexplore.ieee.org/abstract/document/7872466?casa_token=zVfSSrNDko0AAAAA:Nx0KmjBqk0bQfG9MoOHqhwxtNEyz7QquQlkb6w-Mj20IJhAcB_tot4HzByRgg03--76FpLBtcxk
    - https://academic.oup.com/bioinformatics/article/28/24/3290/244641?login=true
    - http://proceedings.mlr.press/v80/yoon18b.html
    - https://arxiv.org/abs/2010.07414
    - https://academic.oup.com/bioinformatics/article-abstract/36/15/4331/5838186?redirectedFrom=fulltext
    - https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-016-0281-4?utm_campaign=BMCF_TrendMD_2020_GenomeMedicine&utm_source=TrendMD&utm_medium=cpc
    - https://www.nature.com/articles/s41467-019-08746-5
    - https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033034

    I will do another search using one of the keywords (data integration) from one of these articles. Already gave a few possible relevant papers:
    - https://academic.oup.com/jamia/article/25/1/99/3826530?login=true
    - https://academic.oup.com/bioinformatics/article/29/14/1830/232698?login=true


# Method Evaluation
Show behaviour of model on artificial datasets when varying  the scale, and density. Sum the heatmap matrix (and normalize). This way different densities and sizes can be compared.

- How to determine range of size and density to evaluate?


# What does the data look like?
Data is a gene regulatory network, similar to protein interaction networks. Examples of protein networks (taken from barabasi paper):
- Metabolic E Coli, Jeong et al 2000
- Protein, S. cerev, Mason et al 2000

## TODO
- Read Reka Albert paper https://jcs.biologists.org/content/118/21/4947
- Read Architecture paper https://link.springer.com/chapter/10.1007/978-0-387-33532-2_5

