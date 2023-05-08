from torch import nn
import torch


class GNLayer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=1, act_fn=nn.SiLU()):
        super(GNLayer, self).__init__()
        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

    def edge_model(self, source, target, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(self, h, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], edge_attr)
        h = self.node_model(h, edge_index, edge_feat)
        return h


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


# def unsorted_segment_mean(data, segment_ids, num_segments):
#     result_shape = (num_segments, data.size(1))
#     segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
#     result = data.new_full(result_shape, 0)  # Init empty result tensor.
#     count = data.new_full(result_shape, 0)
#     result.scatter_add_(0, segment_ids, data)
#     count.scatter_add_(0, segment_ids, torch.ones_like(data))
#     return result / count.clamp(min=1)


class IntroGNLayer(nn.Module):
    def __init__(self, output_nf, hidden_nf, edges_in_d=1, act_fn=nn.SiLU()):
        super(IntroGNLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

    def edge_model(self, edge_attr):
        out = self.edge_mlp(edge_attr)
        return out

    def node_model(self, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=edge_index.max()+1)
        out = self.node_mlp(agg)
        return out

    def forward(self, edge_index, edge_attr):
        edge_feat = self.edge_model(edge_attr)
        h = self.node_model(edge_index, edge_feat)
        return h
