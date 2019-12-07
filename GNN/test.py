import dgl
import networkx
import torch
import numpy as np
from model import GNN, EMB_SIZE

model = GNN(1)

# build a fake graph. Copying from other code
g = dgl.DGLGraph()
# add 34 nodes into the graph; nodes are labeled from 0~33
g.add_nodes(34)
# all 78 edges as a list of tuples
edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
             (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
             (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
             (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
             (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
             (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
             (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
             (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
             (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
             (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
             (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
             (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
             (33, 31), (33, 32)]
# add edges two lists of nodes: src and dst
src, dst = tuple(zip(*edge_list))
g.add_edges(src, dst, {'l_e': torch.ones(78)})
g.add_edges(dst, src, {'l_e': torch.zeros(78)})

e_1 = torch.from_numpy(np.arange(0, 78))
e_0 = torch.from_numpy(np.arange(78, 156))

g.ndata['x_v'] = torch.rand(34, EMB_SIZE)
g.ndata['h_v'] = torch.zeros(34, EMB_SIZE)
g.edata['h_e'] = torch.zeros(156, EMB_SIZE)

# # a smaller graph.
# g.add_nodes(4)
# edge_list = [(1, 0), (2, 0), (2, 1), (3, 0)]
# src, dst = tuple(zip(*edge_list))
# g.add_edges(src, dst, {'l_e': torch.ones(4)})
# g.add_edges(dst, src, {'l_e': torch.zeros(4)})
# e_1 = torch.from_numpy(np.arange(0, 4))
# e_0 = torch.from_numpy(np.arange(4, 8))
# g.ndata['x_v'] = torch.rand(4, EMB_SIZE)
# g.ndata['h_v'] = torch.zeros(4, EMB_SIZE)
# g.edata['h_e'] = torch.zeros(8, EMB_SIZE)

# pass it through.
g_1 = model.forward(e_1, e_0, g)
