import dgl
import torch
import numpy as np
from GNN.model import EMB_SIZE
from GNN.model import ASTactic_EMB_SIZE
from lark.tree import Tree
from torch.nn.parameter import Parameter


def tree_to_graph(ast, device, nonterminal_embs):
    """
    convert an ast Tree to a dgl graph.
    :param ast: an ast Tree
           nonterminal_embs: embeddings for non-terminals
    :return: e_1, e_0: indices of edges with label 1 and 0 respectively.
             g : the dgl graph converted from ast.
    """
    if not isinstance(ast, Tree):
        return

    g = dgl.DGLGraph()
    edge_list = []
    x_v = []

    # traverse the tree, add node and produce edge list.
    def traverse_postorder(node):
        node_idx = g.number_of_nodes()
        g.add_nodes(1)
        x_v.append(nonterminal_embs[node.data])
        for c in node.children:
            if isinstance(c, Tree):
                child_idx = traverse_postorder(c)
                edge_list.append((node_idx, child_idx))
        return node_idx

    traverse_postorder(ast)

    num_nodes = g.number_of_nodes()
    num_edges = len(edge_list)

    # initialize node features.
    g.ndata['x_v'] = torch.stack(x_v, dim=0)
    g.ndata['h_v'] = torch.zeros(num_nodes, EMB_SIZE).to(device)

    # case where there's only one node in the ast
    if num_edges == 0:
        return [], [], g

    # add edges and labels.
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst, {'l_e': torch.ones(num_edges).to(device)})
    g.add_edges(dst, src, {'l_e': torch.zeros(num_edges).to(device)})

    # initialize edge features.
    g.edata['h_e'] = torch.zeros(g.number_of_edges(), EMB_SIZE).to(device)

    # list out edges with the same label.
    e_1 = torch.from_numpy(np.arange(0, num_edges))
    e_0 = torch.from_numpy(np.arange(num_edges, 2*num_edges))

    return e_1, e_0, g


def create_nonterminal_embs(device):
    """
    create embeddings for each non-terminal
    """
    nonterminals = [
        'constr__constr',
        'constructor_rel',
        'constructor_var',
        'constructor_meta',
        'constructor_evar',
        'constructor_sort',
        'constructor_cast',
        'constructor_prod',
        'constructor_lambda',
        'constructor_letin',
        'constructor_app',
        'constructor_const',
        'constructor_ind',
        'constructor_construct',
        'constructor_case',
        'constructor_fix',
        'constructor_cofix',
        'constructor_proj',
        'constructor_ser_evar',
        'constructor_prop',
        'constructor_set',
        'constructor_type',
        'constructor_ulevel',
        'constructor_vmcast',
        'constructor_nativecast',
        'constructor_defaultcast',
        'constructor_revertcast',
        'constructor_anonymous',
        'constructor_name',
        'constructor_constant',
        'constructor_mpfile',
        'constructor_mpbound',
        'constructor_mpdot',
        'constructor_dirpath',
        'constructor_mbid',
        'constructor_instance',
        'constructor_mutind',
        'constructor_letstyle',
        'constructor_ifstyle',
        'constructor_letpatternstyle',
        'constructor_matchstyle',
        'constructor_regularstyle',
        'constructor_projection',
        'bool',
        'int',
        'names__label__t',
        'constr__case_printing',
        'univ__universe__t',
        'constr__pexistential___constr__constr',
        'names__inductive',
        'constr__case_info',
        'names__constructor',
        'constr__prec_declaration___constr__constr____constr__constr',
        'constr__pfixpoint___constr__constr____constr__constr',
        'constr__pcofixpoint___constr__constr____constr__constr',
    ]

    nonterminal_embs = {}

    for obj in nonterminals:
        nonterminal_embs[obj] = Parameter(torch.rand(EMB_SIZE)).to(device)

    return nonterminal_embs

if __name__ == '__main__':
    v = Tree('root', [Tree('leaf_1', []), Tree('leaf_2', []), Tree('leaf_3', [])])
    t = Tree('root', [v, Tree('leaf_1', [])])
    g = tree_to_graph(t)

