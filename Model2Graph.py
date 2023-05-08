import torch

def getEdgesQuickly(model):
    params = list(model.parameters())
    num_nodes = 0
    num_edges = 0
    for i in range(0, len(params), 2):
        weight = params[i]
        outdim, indim = weight.shape
        if i == 0:
            num_nodes += indim
        num_nodes += outdim + 1
        num_edges += outdim * indim + outdim
    edge_index = torch.zeros((2, num_edges), dtype=torch.int64)
    edge_weight = torch.zeros((num_edges,))
    num_nodes = 0
    num_edges = 0
    for i in range(0, len(params), 2):
        weight = params[i]
        outdim, indim = weight.shape
        bias = params[i+1]
        if i == 0:
            # bemeneti réteg csúcsainak hozzáadása
            inputs = torch.arange(indim)
            num_nodes += indim
        # bemenő csúcsok (korábban hozzáadtuk már)
        in_nodes = torch.arange(num_nodes - indim, num_nodes)
        # a bias-hoz tartozó csúcs hozzáadása
        bias_node = torch.arange(num_nodes, num_nodes + 1)
        num_nodes += 1
        # kimenő csúcsok hozzáadása
        out_nodes = torch.arange(num_nodes, num_nodes + outdim)
        num_nodes += outdim
        # összekötés az előző réteggel
        edge_index[:, num_edges:num_edges+outdim*indim] = torch.dstack(torch.meshgrid(in_nodes, out_nodes)).reshape(-1, 2).T
        edge_weight[num_edges:num_edges+outdim*indim] = weight.flatten()
        num_edges += outdim * indim
        # összekötés a bias-hoz tartozó csúccsal
        edge_index[:, num_edges:num_edges+outdim] = torch.dstack(torch.meshgrid(bias_node, out_nodes)).reshape(-1, 2).T
        edge_weight[num_edges:num_edges+outdim] = bias
        num_edges += outdim
    return edge_index, edge_weight, num_nodes