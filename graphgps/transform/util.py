import torch
import numpy as np
import networkx as nx
import graphgps.transform.util


def dict2tensor(dictionary: dict):
    output_list = []
    for _, val in dictionary.items():
        output_list.append(val)
    output = np.stack(output_list)
    output = torch.tensor(output).reshape(-1, 1)
    return output

def get_degree(G: nx.Graph):
    degree_dist = np.asarray([item[1] / 2 for item in sorted(G.degree())]).reshape(-1, 1)
    return torch.tensor(degree_dist)

def get_pagerank(G: nx.Graph):
    pagerank_dist = np.asarray([float(pagerank) for _, pagerank in nx.pagerank(G).items()]).reshape(-1, 1)
    return torch.tensor(pagerank_dist)

def get_closeness_centrality(G: nx.graph):
    closeness_centrality = nx.closeness_centrality(G)
    return dict2tensor(closeness_centrality)
    

def get_local_clustering(G: nx.Graph):
    simple_graph = nx.Graph(G)
    local_clustering_coefficient = nx.clustering(simple_graph)
    return dict2tensor(local_clustering_coefficient)

def get_betweenness_centrality(G: nx.Graph):
    pass


def get_graph_encoding(nx_graph, graph_invariant):
    encoding_list = []
    for name in graph_invariant:
        graph_invariant_func = getattr(graphgps.transform.util, "get_" + name)
        graph_invariant_dist = graph_invariant_func(nx_graph)
        encoding_list.append(graph_invariant_dist)
    graph_encoding = torch.cat(encoding_list, -1)
    # graph_encoding = torch.nn.functional.normalize(graph_encoding, dim=0)
    return graph_encoding
