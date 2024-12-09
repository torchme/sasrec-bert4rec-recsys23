import argparse
import json

import numpy as np
import umap
from sklearn.decomposition import PCA


def reduce_dim(params):
    def pca_reducer(mtrx, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(mtrx)
        reduced_emb = pca.transform(mtrx)
        return reduced_emb

    def umap_reducer(mtrx, n_components):
        umap_reducer = umap.UMAP(n_components)
        reduced_emb = umap_reducer.fit_transform(mtrx)
        return reduced_emb

    n_components = params['n_components']
    embeddings_path = params['embeddings_path']
    save_path = params['save_path']
    reduce_fn = umap_reducer if params['reducer'] == 'UMAP' else pca_reducer

    with open(embeddings_path, 'r') as f:
        user2embs = json.load(f)

    users = list(user2embs.keys())
    embs = np.array(user2embs.values())
    reduced_embs = reduce_fn(embs, n_components)
    new_dct = dict(zip(users, reduced_embs))

    with open(save_path, 'w') as f:
        json.dump(new_dct, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a configuration path.")
    parser.add_argument('config_path', type=str, help='Path to the configuration file')

    args = parser.parse_args()
    print(f"The configuration path is: {args.config_path}")
    with open(args.config_path, 'r') as f:
        params_all = json.load(f)
    for key in params_all:
        reduce_dim(params_all[key])