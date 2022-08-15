import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time
from layer import *
import sys
from collections import OrderedDict
from torch.distributions import Bernoulli

class DGCRN(nn.Module):
    def __init__(self,
                 gcn_depth,
                 num_nodes,
                 device,
                 predefined_A=None,
                 dropout=0.3,
                 subgraph_size=20,
                 node_dim=40,
                 middle_dim=2,
                 seq_length=3,
                 in_dim=2,
                 out_dim=12,
                 layers=3,
                 list_weight=[0.05, 0.95, 0.95],
                 tanhalpha=3,
                 cl_decay_steps=4000,
                 rnn_size=60,
                 hyperGNN_dim=16):
        super(DGCRN, self).__init__()
        self.output_dim = 1

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A

        self.seq_length = seq_length

        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

        self.idx = torch.arange(self.num_nodes).to(device)

        self.rnn_size = rnn_size
        self.in_dim = in_dim

        self.hidden_size = self.rnn_size

        dims_hyper = [
            self.hidden_size + in_dim, hyperGNN_dim, middle_dim, node_dim
        ]

        self.GCN1_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                           'hyper')

        self.GCN2_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                           'hyper')

        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                              'hyper')

        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                              'hyper')

        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                             'hyper')

        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                             'hyper')

        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                                'hyper')

        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                                'hyper')

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        self.alpha = tanhalpha
        self.device = device
        self.k = subgraph_size
        dims = [in_dim + self.hidden_size, self.hidden_size]

        self.gz1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')

        self.gz1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.fc = nn.Linear(self.num_nodes*self.hidden_size+1, self.num_nodes)
        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth

    def preprocessing(self, adj, predefined_A):
        adj = adj + torch.eye(self.num_nodes).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]
    def Controller(self, x,epsilon):
        probs = torch.sigmoid(self.fc(x))
        probs = (1-epsilon)*probs + epsilon*torch.FloatTensor([0.05]).cuda(0)  # Explore/exploit
        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action
        log_pi = m.log_prob(action) # compute log probability of sampled action
        return action.squeeze(0).cuda(0), log_pi.squeeze(0).cuda(0), -torch.log(probs).squeeze(0).cuda(0)
    def BaselineNetwork(self,x):
        b = self.fc(x)
        return b
    def step(self,
             input,
             Hidden_State,
             Cell_State,
             predefined_A,
             type='encoder',
             idx=None,
             i=None):

        x = input
      
        x = x.transpose(1, 2).contiguous()

        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)
        
        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.num_nodes, self.hidden_size)), 2)

        if type == 'encoder':

            filter1 = self.GCN1_tg(hyper_input,
                                   predefined_A[0]) + self.GCN1_tg_1(
                                       hyper_input, predefined_A[1])
            filter2 = self.GCN2_tg(hyper_input,
                                   predefined_A[0]) + self.GCN2_tg_1(
                                       hyper_input, predefined_A[1])

        if type == 'decoder':

            filter1 = self.GCN1_tg_de(hyper_input,
                                      predefined_A[0]) + self.GCN1_tg_de_1(
                                          hyper_input, predefined_A[1])
            filter2 = self.GCN2_tg_de(hyper_input,
                                      predefined_A[0]) + self.GCN2_tg_de_1(
                                          hyper_input, predefined_A[1])

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))

        adj = F.relu(torch.tanh(self.alpha * a))

        adp = self.preprocessing(adj, predefined_A[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_A[1])

        Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size)

        combined = torch.cat((x, Hidden_State), -1)

        if type == 'encoder':
            z = F.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = F.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))
        elif type == 'decoder':
            z = F.sigmoid(
                self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = F.sigmoid(
                self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(
                self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(
            1 - z, Cell_State)

        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(
            -1, self.hidden_size)

    def forward(self,
                input,
                idx=None,
                ycl=None,
                batches_seen=None,
                task_level=1,
                epsilon=0.0):

        predefined_A = self.predefined_A
        x = input

        batch_size = x.size(0)
        Hidden_State, Cell_State = self.initHidden(batch_size * self.num_nodes,
                                                   self.hidden_size)
        print(Hidden_State.shape)

        outputs = None
        halt_points = torch.zeros((batch_size, self.num_nodes)).cuda(0)
        predictions = torch.zeros((batch_size, self.num_nodes)).cuda(0)
        actions=[]
        baselines=[]
        log_pi=[]
        halt_probs=[]
        print("111111111111",x.shape,ycl.shape)
        for i in range(self.seq_length):
            Hidden_State, Cell_State = self.step(torch.squeeze(x[..., i]).unsqueeze(1),
                                                 Hidden_State, Cell_State,
                                                 predefined_A, 'encoder', idx,
                                                 i)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            # go_symbol=self.transformAttention(x, STE_his, STE_pred)
            go_symbol = torch.zeros((batch_size, self.output_dim, self.num_nodes),
                                    device=self.device)
            timeofday = ycl[:, 1:, :, :]

            decoder_input = go_symbol

            outputs_final = []

            for j in range(task_level-i):
                try:
                    decoder_input = torch.cat([decoder_input, timeofday[..., j]],dim=1)
                    decoder_hidden=Hidden_State
                except:
                    print(decoder_input.shape, timeofday.shape)
                    sys.exit(0)
                decoder_hidden, Cell_State = self.step(decoder_input, decoder_hidden,
                                                    Cell_State, predefined_A,
                                                    'decoder', idx, None)

                decoder_output = self.fc_final(decoder_hidden)

                decoder_input = decoder_output.view(batch_size, self.num_nodes,
                                                    self.output_dim).transpose(
                                                        1, 2)
               
                if self.training and self.use_curriculum_learning:
                    c = np.random.uniform(0, 1)
                    if c < self._compute_sampling_threshold(batches_seen):
                        decoder_input = ycl[:, :1, :, j]

            outputs_final = decoder_output.view(batch_size, self.num_nodes)
            c_in = torch.cat((decoder_hidden.transpose(0,1).unsqueeze(0),torch.tensor([i], dtype=torch.float, requires_grad=False).view(1, 1,1).repeat(1,self.hidden_size, 1).cuda(0)),dim=2)
            a_t, p_t, w_t = self.Controller(c_in,epsilon)
            b_t = self.BaselineNetwork(torch.cat((decoder_hidden.transpose(0,1).unsqueeze(0),torch.tensor([i], dtype=torch.float, requires_grad=False).view(1, 1,1).repeat(1,self.hidden_size, 1).cuda(0)),dim=2).detach())
            
            predictions = torch.where((a_t == 1) & (predictions == 0), outputs_final, predictions)
            halt_points = torch.where((halt_points == 0) & (a_t == 1), torch.tensor([i+1],dtype=torch.float, requires_grad=False).view(1, 1).repeat(self.hidden_size,self.num_nodes).cuda(0), halt_points)
           
            actions.append(a_t)
            baselines.append(b_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == 0).sum() == 0:  # If no negative values, every class has been halted
                break

        predictions = torch.where(predictions == 0, outputs_final, predictions)
        halt_points = torch.where(a_t == 1, torch.tensor([i+1],dtype=torch.float, requires_grad=False).view(1, 1).repeat(self.hidden_size,self.num_nodes).cuda(0), halt_points)

       
        baselines = torch.stack(baselines)
        log_pi = torch.stack(log_pi)
        wait_penalty= torch.stack(halt_probs).sum(0).sum(1).mean() 
        grad_mask = torch.zeros_like(torch.stack(actions).transpose(0, 1))

        for b in range(batch_size):
            for n in range(self.num_nodes):
                grad_mask[b, :(halt_points[b, n]).long(),n] = 1
       
        return predictions, (halt_points).mean()/self.seq_length,grad_mask.transpose(1,0),baselines.float(),log_pi,wait_penalty

    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device))

            nn.init.orthogonal(Hidden_State)
            nn.init.orthogonal(Cell_State)

            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
