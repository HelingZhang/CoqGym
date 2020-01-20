import torch
import torch.nn as nn

H_SIZE_1 = 256 # size of the 1st hidden layer. 256 in the paper.
H_SIZE_2 = 128 # size of the 2nd hidden layer. 128 in the paper.
EMB_SIZE = 128 # size of embeddings. 128 in the paper.

class MLP(nn.Module):
    """
    used for MLP_v and MLP_e in the main model.
    two hidden layers of sizes 256 and 128 with ReLU activations.
    """
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, H_SIZE_1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(H_SIZE_1, H_SIZE_2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(H_SIZE_2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        return out

class GNN(nn.Module):
    """
    the GNN model.
    deals only with connected graphs.
    T - total number of hops.
    """
    def __init__(self, T, device):
        super(GNN, self).__init__()
        self.T = T
        self.device = device
        self.t = None # current time step.
        self.MLP_v = MLP(EMB_SIZE, EMB_SIZE)
        self.MLP_e = MLP(1, EMB_SIZE) # why?
        # self.edge_emb_1 = torch.rand(EMB_SIZE)
        # self.edge_emb_0 = torch.rand(EMB_SIZE)
        self.MLP_edge_1 = nn.ModuleList([MLP(EMB_SIZE*3, EMB_SIZE)]*self.T) # computes message from parent nodes.
        self.MLP_edge_0 = nn.ModuleList([MLP(EMB_SIZE*3, EMB_SIZE)]*self.T) # computes message from child nodes.
        self.MLP_aggr = MLP(EMB_SIZE*3, EMB_SIZE)

    def message_1(self, edges):
        """
        computes the message along an edge whose label is 1.
        edges - an EdgeBatch, representing all edges.
        t - current time step.
        """
        t = self.t
        input = torch.cat((edges.src['h_v'], edges.dst['h_v'], edges.data['h_e']), 1).to(self.device) # input to MLP_edge_1.
        num_edges = input.size(0)
        message = torch.stack([self.MLP_edge_1[t].forward(input[i,:]) for i in range(num_edges)])
        return {'m_1': message}

    def message_0(self, edges):
        """
        computes the message along an edge whose label is 0.
        edges - an EdgeBatch, representing all edges.
        t - current time step.
        """
        t = self.t
        input = torch.cat((edges.src['h_v'], edges.dst['h_v'], edges.data['h_e']), 1).to(self.device) # input to MLP_edge_0.
        num_edges = input.size(0)
        message = torch.stack([self.MLP_edge_0[t].forward(input[i, :]) for i in range(num_edges)])
        return {'m_0': message}

    def update(self, nodes):
        """
        the update function.
        nodes - a NodeBatch, representation all nodes.
        t - current time step.
        """
        t = self.t
        avg_m_1 = torch.mean(nodes.mailbox['m_1'], dim=1)
        avg_m_0 = torch.mean(nodes.mailbox['m_0'], dim=1)
        h_old = nodes.data['h_v'].to(self.device)
        input = torch.cat((h_old, avg_m_1, avg_m_0), 1)
        avg_m_1_size = avg_m_1.size()
        avg_m_0_size = avg_m_0.size()
        h_old_size = h_old.size()
        input_size = input.size()# input to MLP_aggr.
        h_new = self.MLP_aggr.forward(input)
        h_new = h_new + h_old
        return {'h_v': h_new.to('cpu')}

    def forward(self, e_1, e_0, g):
        """
        g - a directed graph with boolean edge labels.
        e_1 - array of edges labeled as 1.
        e_0 - array of edges labeled as 0.
        """
        for v in range(len(g.nodes())):
            # initialize node embeddings from (the embedding of) node labels.
            x_v = g.nodes[v].data['x_v'].to(self.device)
            g.nodes[v].data['h_v'] = self.MLP_v.forward(x_v).to('cpu')
        for e in range(len(g.edges()[0])):
            # initialize edge embeddings from edge labels.
            l_e = g.edges[e].data['l_e'].to(self.device)
            g.edges[e].data['h_e'] = self.MLP_e.forward(l_e).view(1, -1).to('cpu') # abusing nn?
            # g.edges[e].data['h_e'] = self.edge_emb_0.view(1, -1) if l_e == 0 else self.edge_emb_1.view(1, -1)

        # message passing.
        for t in range(self.T):
            self.t = t
            # compute the messages
            g.send(g.find_edges(e_1), self.message_1)
            g.send(g.find_edges(e_0), self.message_0)
            # update according to message.
            g.recv(g.nodes(), self.update)

        # readout
        # diverges from original paper.
        # TODO: implement readout. resolve possible incompatibality due to output dimensions.
        return g


# Note
# structure of g:
#     - for each node v:
#         - x_v: embedding of its 'token'
#         - h_v: node embeddings
#     - for each edge e:
#         - l_e: 0 or 1.
#         - h_e: embedding derived from 0 or 1. The paper derived this by passing this trough a MLP. Why not just make trainable embeddings for 0 and 1? Wired.