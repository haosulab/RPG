import math

import numpy as np
import torch.nn as nn
from torch import nn as nn
from torch.nn import Parameter, functional as F

from torch.nn import Module
from torch.nn import LayerNorm
import torch


import inspect
def save_init_params(self, locals_):
    spec = inspect.getfullargspec(self.__init__)
    if spec.varkw:
        kwargs = locals_[spec.varkw].copy()
    else:
        kwargs = dict()
    if spec.kwonlyargs:
        for key in spec.kwonlyargs:
            kwargs[key] = locals_[key]
    if spec.varargs:
        varargs = locals_[spec.varargs]
    else:
        varargs = tuple()
    in_order_args = [locals_[arg] for arg in spec.args][1:]
    self.__args = tuple(in_order_args) + varargs
    self.__kwargs = kwargs
    setattr(self, "_serializable_initialized", True)



class Attention(Module):
    """
    Additive, multi-headed attention
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 activation_fnx=F.leaky_relu,
                 softmax_temperature=1.0):
        save_init_params(self, locals())
        super().__init__()
        #assert num_heads == 1

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        if self.num_heads == 1:
            self.fc_createheads = nn.Linear(embedding_dim, embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(embedding_dim, embedding_dim)
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))
        self.activation_fnx = activation_fnx

    def forward(self, query, context, memory):
        """
        N, nV, nE memory -> N, nV, nE updated memory
        :param query:
        :param context:
        :param memory:
        :return:
        """
        N, nQ, nE = query.size()
        nV = memory.size(1)
        if self.num_heads == 1:
            #query = self.fc_createheads(query).view(N, nQ, nE)
            #return torch.ones((N, nQ, nE), device=query.device, dtype=query.dtype)
            query = query.unsqueeze(2).expand(-1, -1, nV, -1)
            context = context.unsqueeze(1).expand_as(query)
            qc_logits = self.fc_logit(torch.tanh(context + query)).squeeze(-1)
            attention_probs = F.softmax(qc_logits / self.softmax_temperature, dim=2)
            attention_heads = self.activation_fnx(torch.bmm(attention_probs, memory))
            attention_result = self.fc_reduceheads(attention_heads.view(N, nQ, nE))
        else:
            # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
            #print(context.shape, query.shape, memory.shape)
            k = context.view(N, -1, self.num_heads, self.embedding_dim//self.num_heads)
            q = query.view(N, -1, self.num_heads, self.embedding_dim//self.num_heads)
            v = memory.view(N, -1, self.num_heads, self.embedding_dim//self.num_heads)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / self.softmax_temperature
            scores = F.softmax(scores, dim=-1)
            output = self.activation_fnx(torch.matmul(scores, v))
            concat = output.transpose(1, 2).contiguous().view(N, -1, self.embedding_dim)
            attention_result = self.fc_reduceheads(concat)

        return attention_result


class AttentiveGraphToGraph(Module):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True,
                 **kwargs):
        save_init_params(self, locals())
        super().__init__()
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attention = Attention(embedding_dim, num_heads=num_heads)
        #self.layer_norm= nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, vertices):
        """
        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        assert len(vertices.size()) == 3

        # -> (N, nQ, nE), (N, nV, nE), (N, nV, nE)

        # if self.layer_norm is not None:
        #     qcm_block = self.layer_norm(self.fc_qcm(vertices))
        # else:
        qcm_block = self.fc_qcm(vertices)

        query, context, memory = qcm_block.chunk(3, dim=-1)

        return self.attention(query, context, memory)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def identity(x):
    return x

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class AttentiveGraphPooling(Module):
    """
    Pools nV vertices to a single vertex embedding
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 init_w=3e-3,
                 layer_norm=True,
                 mlp_kwargs=None):
        save_init_params(self, locals())
        super().__init__()
        self.num_heads = num_heads
        self.fc_cm = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        self.input_independent_query = Parameter(torch.Tensor(embedding_dim))
        self.input_independent_query.data.uniform_(-init_w, init_w)
        # self.num_heads = num_heads
        #assert num_heads == 1
        self.attention = Attention(embedding_dim, num_heads=num_heads)

        if mlp_kwargs is not None:
            self.proj = Mlp(**mlp_kwargs)
        else:
            self.proj = None

    def forward(self, vertices):
        """
        N, nV, nE -> N, nE
        :param vertices:
        :return:
        """
        N, nV, nE = vertices.size()

        # nE -> N, nQ, nE where nQ == self.num_heads
        query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)

        # if self.layer_norm is not None:
        #     cm_block = self.layer_norm(self.fc_cm(vertices))
        # else:
        # cm_block = self.fc_cm(vertices)
        # context, memory = cm_block.chunk(2, dim=-1)
        if self.num_heads == 1:
            context = vertices
            memory = vertices
        else:
            #print(vertices.shape)
            cm_block = self.fc_cm(vertices)
            context, memory = cm_block.chunk(2, dim=-1)

        # gt.stamp("Readout_preattention")
        attention_result = self.attention(query, context, memory)

        # gt.stamp("Readout_postattention")
        # return attention_result.sum(dim=1) # Squeeze nV dimension so that subsequent projection function does not have a useless 1 dimension
        if self.proj is not None:
            return self.proj(attention_result).squeeze(1)
        else:
            return attention_result

            
            

class GraphPropagation(Module):
    """
    Input: state
    Output: context vector
    """

    def __init__(self,
                 num_relational_blocks=1,
                 num_query_heads=1,
                 graph_module_kwargs=None,
                 layer_norm=False,
                 activation_fnx=F.leaky_relu,
                 graph_module=AttentiveGraphToGraph,
                 post_residual_activation=True,
                 recurrent_graph=False,
                 **kwargs
                 ):
        """
        :param embedding_dim:
        :param lstm_cell_class:
        :param lstm_num_layers:
        :param graph_module_kwargs:
        :param style: OSIL or relational inductive bias.
        """
        save_init_params(self, locals())
        super().__init__()

        # Instance settings

        self.num_query_heads = num_query_heads
        self.num_relational_blocks = num_relational_blocks
        assert graph_module_kwargs, graph_module_kwargs
        self.embedding_dim = graph_module_kwargs['embedding_dim']

        if recurrent_graph:
            rg = graph_module(**graph_module_kwargs)
            self.graph_module_list = nn.ModuleList(
                [rg for i in range(num_relational_blocks)])
        else:
            self.graph_module_list = nn.ModuleList(
                [graph_module(**graph_module_kwargs) for i in range(num_relational_blocks)])

        # Layer norm takes in N x nB x nE and normalizes
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for i in range(num_relational_blocks)])

        # What's key here is we never use the num_objects in the init,
        # which means we can change it as we like for later.

        """
        ReNN Arguments
        """
        self.layer_norm = layer_norm
        self.activation_fnx = activation_fnx

    def forward(self, vertices, *kwargs):
        """
        :param shared_state: state that should be broadcasted along nB dimension. N * (nR + nB * nF)
        :param object_and_goal_state: individual objects
        :return:
        """
        output = vertices

        for i in range(self.num_relational_blocks):
            new_output = self.graph_module_list[i](output)
            new_output = output + new_output

            output = self.activation_fnx(new_output) # Diff from 7/22
            # Apply layer normalization
            if self.layer_norm:
                output = self.layer_norms[i](output)
        return output