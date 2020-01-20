import torch
from GNN.model import GNN
from GNN.utils import tree_to_graph

class TermEncoder(nn.Module):
    """
    term encoder based on GNN instead of TreeLSTM.
    to replace the TermEncoder class in ASTactic.term_encoder.
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.model = GNN(2, opts.device)

    def forward(self, term_asts):
        """
        term_asts: a list of asts
        """
        graph_embs = []
        for ast in term_asts:
            e_1, e_0, g = tree_to_graph(ast)
            graph_embs.append(self.model.forward(e_1, e_0, g))

        return torch.stack(graph_embs)