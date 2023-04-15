import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering

# Read the data from CSV file
df = pd.read_csv('final_dataset.csv')

# Create a bipartite graph
B = nx.Graph()
for _, row in df.iterrows():
    B.add_node(row['country'], bipartite=0)
    B.add_node(row['organization'], bipartite=1)
    B.add_node(row['attack_type'], bipartite=2)
    B.add_edge(row['country'], row['organization'])
    B.add_edge(row['organization'], row['attack_type'])

# Get the adjacency matrix and perform spectral clustering
adj_matrix = nx.adjacency_matrix(B)
spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
labels = spectral.fit_predict(adj_matrix)

# Get the list of countries and organizations
countries = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
orgs = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

# Make a dataframe with all possible pairs of countries and organizations
df_pairs = pd.DataFrame([(c, o) for c in countries for o in orgs], columns=['country', 'organization'])

# Compute the similarity matrix using spectral curve fitting
def similarity(x, y):
    return 1 / (1 + np.exp(-(x - y)))

sim_matrix = np.zeros((len(countries), len(orgs)))
for i, c in enumerate(countries):
    for j, o in enumerate(orgs):
        if labels[i] != labels[j+len(countries)]:
            sim_matrix[i, j] = similarity(1, 0)
        else:
            neighbors_c = [n for n in B.neighbors(c)]
            neighbors_o = [n for n in B.neighbors(o)]
            num_common_neighbors = len(set(neighbors_c).intersection(set(neighbors_o)))
            sim_matrix[i, j] = similarity(num_common_neighbors, 2)

# Predict the links using the similarity matrix
pred_links = []
for i, c in enumerate(countries):
    for j, o in enumerate(orgs):
        if sim_matrix[i, j] > 0.5:
            pred_links.append((c, o))

# Make a dataframe with the predicted links and output to CSV
df_pred = pd.DataFrame(pred_links, columns=['country', 'organization'])
df_pred.to_csv('final_predicted_links.csv', index=False)
