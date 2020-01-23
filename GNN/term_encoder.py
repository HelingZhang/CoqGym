import torch
import torch.nn as nn
from GNN.model import GNN
from GNN.utils import *


class TermEncoder(nn.Module):
    """
    term encoder based on GNN instead of TreeLSTM.
    to replace the TermEncoder class in ASTactic.term_encoder.
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.model = GNN(2, opts.device)
        self.nonterminal_embs = create_nonterminal_embs(opts.device)

    def forward(self, term_asts):
        """
        term_asts: a list of asts
        """
        # print(len(term_asts))
        graph_embs = []
        for ast in term_asts:
            e_1, e_0, g = tree_to_graph(ast, self.opts.device, self.nonterminal_embs)
            # #############
            # if g is None:
            #     graph_embs.append(e_1)
            #     continue
            # ##############3
            emb = self.model.forward(e_1, e_0, g)
            graph_embs.append(emb)

        return torch.stack(graph_embs)