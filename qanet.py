"""QANet layers implementation

Authors:
    Louise Huang
    Jeremy Mi Yu
    Carson Yu Tian Zhao

Citation:
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    - classes: Multihead Attention, Positional Encoding
    - functions: clones, attention
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import math, copy, time
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

from layers import RNNEncoder

### start our code:
# layers in the QANet model

def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderBlock(nn.Module):
    """Encoder block layer from QANet

    inputs:
        - input_size
        - kernel_size
        - filters
        - num_conv_layers
        - drop_prob

    """
    def __init__(self, input_size, kernel_size, filters, num_conv_layers, drop_prob):
        super(EncoderBlock, self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.num_conv_layers = num_conv_layers
        self.drop_prob = drop_prob

        # positional encoding
        self.pe = PositionalEncoding(d_model=self.input_size, drop_prob=self.drop_prob, max_len=700)
                    # max_len: args.para_limit
                    # crashes if greater than max_len

        # convolutional layer
        self.convLayer = nn.ModuleList([ConvSubLayer(self.input_size, self.kernel_size,
                                    self.filters, self.drop_prob) for _ in range(num_conv_layers)] )

        # Transformers
        self.transform = Transformers(self.input_size, self.kernel_size, self.filters, self.drop_prob)


    def forward(self, x, mask=None):
        # each operation is placed inside a residual block
        # for input x and given operation f, the output is f(layernorm(x) + x)

        # positional encoding
        x = self.pe(x)

        # convolution section
        for i, layer in enumerate(self.convLayer):
            if i % 2 == 0:
                x = F.dropout(x, self.drop_prob, self.training)

            x2 = layer(x)          # output x is f(layernorm(x)) + x
            x = layer_dropout(input = x2, resid = x, training = self.training)

        # attention section
        out = self.transform(x, mask)   # output is after feedforward layer in encoder block

        return out

def layer_dropout(input, resid, training=False, drop_prob=0.95):
    """Helper function for stochastic layer dropout"""
    if training:
        if torch.rand(1) > drop_prob:
            out = F.dropout(input, p=0.1, training=training)
            return out + resid
        else:
            return resid
    else:
        return input + resid

class ConvSubLayer(nn.Module):
    """Convolutional sub-layer in Encoder Block from QANet

    inputs:
        - input_size
        - kernel_size
        - filters
        - num_conv_layers
        - drop_prob

    """
    def __init__(self, input_size, kernel_size, filters, drop_prob):
        super(ConvSubLayer, self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.drop_prob = drop_prob

        # Layernorm
        self.normConv = nn.LayerNorm(normalized_shape = self.input_size,
                                eps=1e-05, elementwise_affine=True)

        # Convolutional sublayer
        self.depthwise = nn.Conv1d(self.input_size, self.input_size,
                                   kernel_size=kernel_size, padding=int((kernel_size-1)/2),
                                   groups=self.input_size)
        self.separable = nn.Conv1d(self.input_size, self.input_size, kernel_size=1)


        # self.convLayer = nn.Conv1d(self.input_size, self.input_size, self.kernel_size, bias = True)
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                # stride=1, padding=0, dilation=1, groups=1, bias=True)

        # self.maxPool = nn.MaxPool1d(self.filters)
        # torch.nn.MaxPool1d(kernel_size, stride=None, padding=0,
                # dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):

        # f(layernorm(x)) + x
        # print("input to ConvSubLayer: ", x.size())   # [10, 150, 128]
        lay_norm_x = self.normConv(x)
        lay_norm_x = lay_norm_x.permute(0, 2, 1)    # [10, 512, 150]

        lay_norm_x = self.depthwise(lay_norm_x)  # [10, 128, 150]
        out = self.separable(lay_norm_x)   # [10, 128, 150]


        # out = self.convLayer(lay_norm_x)            # out: [10, 128, 144]   # 150 - 7 + 1
        # print("convLayer: ", out.size())
        # out = self.maxPool(out)                     # [10, 128, 1]
        # print("maxPool: ", out.size())


        out = out.permute(0, 2, 1)                  # [10, 150, 128]
        return x + out     # [10, 150, 128] + [10, 150, 128]



class Transformers(nn.Module):
    """ Inspired by "All You Need is Attention"
    - Modified, without Decoder Stack (unnecessary for QANet)
    - layernorm, self-attention, layernorm, feedforward

    inputs:
        input_size
        kernel_size
        filters,
        drop_prob

    """
    def __init__(self, input_size, kernel_size, filters, drop_prob):
        super(Transformers, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.drop_prob = drop_prob

        # initialize Layernorms
        self.normAtt = nn.LayerNorm(normalized_shape = self.input_size, eps=1e-05, elementwise_affine=True)
        self.normFF = nn.LayerNorm(normalized_shape = self.input_size, eps=1e-05, elementwise_affine=True)

        # initialize self-attention
        self.multiHeadAtt = MultiHeadAtt(num_heads = 8, d_model = self.input_size, drop_prob=0.1)

        # initialize feed forward
        self.feedforward1 = nn.Linear(in_features = self.input_size,
                                      out_features = 4*self.input_size, bias=True)
        self.feedforward2 = nn.Linear(in_features = 4*self.input_size,
                                      out_features = self.input_size, bias=True)


    def forward(self, x, mask=None):
        """
        @input: the output of the Convolution sub-layer
        @returns: fully connected feed-forward network sublayer output
        """
        # apply dropout during training at every layer just before adding residual (self.training method)
        # x = F.dropout(x, self.drop_prob, self.training)

        # f(layernorm(x)) + x
        x_norm = self.normAtt(x)

        # self-attention
        selfatt_out = self.multiHeadAtt(query = x_norm, key = x_norm, value = x_norm, mask=mask)
            # what to do with mask??
        selfatt_out = layer_dropout(input = selfatt_out, resid = x, training=self.training)

        # f(layernorm(x)) + x
        inFF = self.normFF(selfatt_out)

        # fully connected feed-forward network (with ReLU in between)
        outFF1 = F.relu(self.feedforward1(inFF))
        outFF1 = F.dropout(outFF1, self.drop_prob, self.training)
        outFF = self.feedforward2(outFF1)
        outFF = layer_dropout(input = outFF, resid = selfatt_out, training=self.training)

        return outFF


def attention(query, key, value, mask=None, dropout=None):
    """
    Helper function to compute Scaled Dot Product Attention
    citation: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class MultiHeadAtt(nn.Module):
    """Implement Multihead Attention as specified in "All You Need is Attention"

    inputs:
        - num_heads
        - d_model
        - drop_prob

    citation: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, num_heads, d_model, drop_prob=0.1):
        """Take in number of heads and model size"""
        super(MultiHeadAtt, self).__init__()
        assert d_model % num_heads == 0
        # assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.h = num_heads

        # nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=drop_prob)


    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        # mask = None
        if mask is not None:
            # print("mask size", mask.size())  # [10, 150]
            # Is this even right?? We just hacked our way to fit the dimensions

            # Same mask applied to all h heads.
            # mask = mask.unsqueeze(1)   # original
            mask = mask.unsqueeze(-1)    # [10, 150, 1]
            mask = mask.unsqueeze(0)
            mask = mask.expand(self.h, query.shape[0], query.shape[1], query.shape[1]).permute(1,0,2,3)
            # print("mask size expanded", mask.size())
            # [10, 8, 150, 150]

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.

        # want: [10, 8, 150, 150]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    """
    Implement the Positional Encoding function
    citation: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, d_model, drop_prob, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size, drop_prob):
        super(QANetOutput, self).__init__()
        self.W1 = nn.Linear(2*input_size, 1, bias=False)
        self.W2 = nn.Linear(2*input_size, 1, bias=False)

        # self.rnn = RNNEncoder(input_size=2*input_size,
        #                       hidden_size=input_size,
        #                       num_layers=1,
        #                       drop_prob=drop_prob)

        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)


    def forward(self, m0, m1, m2, mask):

        # Input Shapes: (batch_size, seq_len, input_size)

        M01 = torch.cat((m0, m1), dim = 2)
        M02 = torch.cat((m0, m2), dim = 2)
        logits_1 = self.W1(M01)
        # M02 = self.rnn(M02, mask.sum(-1))
        logits_2 = self.W2(M02)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2




### end our code



