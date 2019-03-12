"""Top-level model classes.

Authors:
    Louis Miao Miao Huang
    Jeremy Mi Yu
    Carson Yu Tian Zhao
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

import qanet as qa
import transformers as trf
import copy


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, char_vectors, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    char_vectors = char_vectors)   # added last line

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)


        ### start our code:
        self.selfattention = layers.SelfAttention(input_size = 8 * hidden_size,
                                                  hidden_size=hidden_size,
                                                  dropout = 0.2)

        ### end our code
        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    # def forward(self, cw_idxs, qw_idxs):    # orig
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)


        # Word/Char embeddings layer
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # RNN encoder layer
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # BiDAF Attention layer
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # BiDAF modeling layer
        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        ### start our code:
        # Self-attention layer
        mod = self.selfattention(mod)

        ### end our code

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


### start our code:
class QANet(nn.Module):
    """Implementation of QANet:
    Combining local convolution with global self-attention

    Structure:
        - Input embedding layer
        - Embedding encoder layer
        - Context-Query attention layer
        - Model encoder layer
        - Output layer

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        char_vectors (torch.Tensor):
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, kernel_size, filters, drop_prob=0.):
        super(QANet, self).__init__()

        # Input embedding layer
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    char_vectors=char_vectors)   # added character vectors

        # resize input embedding layer output size to fit embedding encoder layer input size
        self.resize_emb_pe = nn.Linear(in_features = hidden_size, out_features = filters, bias=False)

        # Embedding encoder layer
        self.emb_enc = qa.EncoderBlock(input_size=filters, kernel_size=kernel_size,
                                       filters=filters, num_conv_layers=4, drop_prob=drop_prob)

        # Context-Query attention layer
        self.att = layers.BiDAFAttention(hidden_size = filters,
                                         drop_prob = drop_prob)

        # Model encoder layer:
        mod_enc = qa.EncoderBlock(input_size=4*filters, kernel_size=5,
                                  filters=filters, num_conv_layers=2, drop_prob=drop_prob)
        self.mod_enc = nn.ModuleList([mod_enc]*7)   # 7 number of blocks

        # QANet Output layer
        self.output = qa.QANetOutput(input_size=4*filters, drop_prob=drop_prob)


    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # Input embedding layer (BiDAF)
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # resize input embedding layer output size to fit embedding encoder layer input size
        c_emb = self.resize_emb_pe(c_emb)   # [batch_size, 150, filters]
        q_emb = self.resize_emb_pe(q_emb)   # [batch_size, 14, filters]

        # Embedding encoder block layer
        c_enc = self.emb_enc(c_emb, c_mask)         # [batch_size, 150, filters]
        q_enc = self.emb_enc(q_emb, q_mask)         # [batch_size, 14, filters]

        # Context-Query attention layer
        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4*filters)

        # Model encoder layer: 7 blocks, 3 repetitions of the model encoder
 

        for mod in self.mod_enc:
            x = mod(att, c_mask)
            # x = mod(att)
        m0 = x
        for mod in self.mod_enc:
            x = mod(x, c_mask)
            # x = mod(x)
        m1 = x
        for mod in self.mod_enc:
            x = mod(x, c_mask)
            # x = mod(x)
        m2 = x


        # Output layer
        out = self.output(m0, m1, m2, c_mask)

        return out



### end our code


# print("QANet output layer out:", out)    # a bunch of way too large negatives (log softmax)
# after 3 epochs:
# tensor([[-1063.9565, -3112.8508, -2586.9385,  ..., -4385.2490, -2570.1309,
#  -3243.5054],
# [-3650.5527, -5642.0811, -4517.4463,  ..., -7321.2461, -4592.7441,
#  -4352.4302],
# [-4758.7959, -4286.9883, -6236.1221,  ..., -4354.2280, -7331.7358,
#  -4595.4023],
# ..., grad_fn=<LogSoftmaxBackward>))

# c_mask: tensor([[1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         ...,
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1]], dtype=torch.uint8)
# q_mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.uint8)

# In comparison, V4 has after 3 epochs:
# (tensor([[-5.0147, -5.0126, -5.0092,  ..., -4.9993, -5.0096, -5.0032],
#     [-5.0058, -5.0155, -5.0099,  ..., -5.0149, -5.0072, -5.0035],
#     [-5.0167, -5.0091, -5.0186,  ..., -5.0129, -4.9903, -5.0170],
#     ...,
#     [-5.0080, -5.0084, -5.0146,  ..., -5.0098, -4.9968, -5.0138],
#     [-5.0083, -5.0028, -5.0196,  ..., -5.0125, -5.0197, -4.9985],
#     [-5.0109, -5.0019, -5.0143,  ..., -5.0129, -5.0025, -5.0012]],
#    grad_fn=<LogSoftmaxBackward>),
