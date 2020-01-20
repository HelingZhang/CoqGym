import dgl
import torch
import numpy as np
from GNN.model import EMB_SIZE
from lark.tree import Tree

def tree_to_graph(ast):
    """
    convert an ast Tree to a dgl graph.
    :param ast: an ast Tree
    :return: e_1, e_0: indices of edges with label 1 and 0 respectively.
             g : the dgl graph converted from ast.
    """
    if not isinstance(ast, Tree):
        return

    g = dgl.DGLGraph()
    edge_list = []

    # traverse the tree, add node and produce edge list.
    def traverse_postorder(node):
        node_idx = g.number_of_nodes()
        g.add_nodes(1)
        for c in node.children:
            if isinstance(c, Tree):
                child_idx = traverse_postorder(c)
                edge_list.append((node_idx, child_idx))
        return node_idx

    traverse_postorder(ast)

    num_nodes = g.number_of_nodes()
    num_edges = len(edge_list)

    # add edges and labels.
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst, {'l_e': torch.ones(num_edges)})
    g.add_edges(dst, src, {'l_e': torch.zeros(num_edges)})

    # initialize edge features.
    g.ndata['x_v'] = torch.rand(num_nodes, EMB_SIZE)
    g.ndata['h_v'] = torch.zeros(num_nodes, EMB_SIZE)
    g.edata['h_e'] = torch.zeros(g.number_of_edges(), EMB_SIZE)

    # list out edges with the same label.
    e_1 = torch.from_numpy(np.arange(0, num_edges))
    e_0 = torch.from_numpy(np.arange(num_edges, 2*num_edges))

    return e_1, e_0, g

if __name__ == '__main__':
    v = Tree('root', [Tree('leaf_1', []), Tree('leaf_2', []), Tree('leaf_3', [])])
    t = Tree('root', [v, Tree('leaf_1', [])])
    g = tree_to_graph(t)

