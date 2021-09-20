import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable

class self_attention_net(nn.Module):
    def __init__(self,  label_hidden_size, max_sen, atten_size):
        super(self_attention_net, self).__init__()
        self.hidden_size = label_hidden_size
        self.sen_len = max_sen
        self.attention_size = atten_size
        self.w_omega = Variable(torch.zeros(self.hidden_size * 2, self.attention_size).cuda())
        self.u_omega = Variable(torch.zeros(self.attention_size).cuda())

    def forward(self, lstm_output):
        # print(lstm_output.size()) = (batch_size,squence_length,  hidden_size*layer_size)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * 2]).cuda()
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega)).cuda()
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1])).cuda()
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sen_len]).cuda()
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1]).cuda()
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sen_len, 1]).cuda()
        attn_output = torch.sum(lstm_output * alphas_reshape, 1).cuda()
        # lstm_output = torch.sum(lstm_output,1)
        # out = torch.cat((attn_output,lstm_output),dim=1)
        #attn_output.size()   num_class*hiddensize*2(双向lstm)
        return attn_output


class sen_attention_label(nn.Module):
    def __init__(self, num_class):
        super(sen_attention_label, self).__init__()
        self.class_num = num_class

    def forward(self, x, label):
        # x.size():(batch_size, hidden_size)  query
        # label:(num_class, label_hidden_size*2)  Xi
        ### 计算方式点积
        label = label.transpose(0, 1).cuda()
        # print('label.size():',label.size())
        m = torch.tanh(torch.mm(x, label)).cuda()
        exps = torch.exp(m).cuda()
        a = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1]).cuda()  ###算权重
        a_reshape = torch.Tensor.reshape(a, [self.class_num, -1]).cuda()
        self.a = a.detach()
        self.a_reshape = a_reshape.detach()
        # print('a_reshape',a_reshape.size())
        # label_attn_output = label_attn_output.transpose(0, 1)
        # print('label_attn_output.size():', label_attn_output.size())
        finalx = torch.mm(label, a_reshape).cuda()
        self.finalx = finalx.detach()
        finalx = finalx.transpose(0, 1).cuda()

        # finalx = torch.mm(x, finalx)
        # print('finalx',finalx.size())

        # lstm_output = torch.sum(lstm_output, 1)
        # out = torch.cat((sen_attn_output, finalx), dim = 1)  #横着拼  (batch_size, hidden_size*layer_size*2)
        # out = torch.cat((x, finalx), dim=1).cuda()
        # out = (x + finalx).cuda()
        # print('out', out.size())
        # lstm_output = torch.sum(lstm_output, 1)
        # output = torch.cat((lstm_output, out), dim = 1)
        return finalx

class label_layer(nn.Module):
    def __init__(self, num_class, embed_dim, label_hidden_size, max_sen, atten_size):
        super(label_layer, self).__init__()
        self.class_num = num_class
        self.embed_dim = embed_dim
        self.hidden_size = label_hidden_size
        self.attention_net = self_attention_net(label_hidden_size,max_sen,atten_size)
        self.lstm = nn.LSTM(embed_dim, label_hidden_size, 1, batch_first=True,
                        dropout=0.5,
                        bidirectional=True)
        self.sen_attention_label = sen_attention_label(num_class=num_class)

    def forward(self, x, label):
        s, b, f = label.size()
        h_0 = Variable(torch.zeros(2, s, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, s, self.hidden_size).cuda())
        label, (final_hidden_state, final_cell_state) = self.lstm(label, (h_0, c_0))
        label = self.attention_net(label)
        out = self.sen_attention_label(x, label)
        return out


################################################################################
####Attention层
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0       ###d_model是embed_dim

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # print("attention", x.size())
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # print("Multihead_size", x.size())
        ####输出应该为 batch_size_*_sen_len_*_embed_size
        return self.output_linear(x)


###Sublayer层
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(features))
    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.a_2 * (x - mean) / (std + self.eps) + self.b_2).cuda()

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, d_model_embed, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        mul_attn = self.dropout(sublayer(self.norm(x))) + x
        # print('SublayerConnection mul_attn', mul_attn.size())
        return mul_attn.cuda()


####Positionwise层
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GELU()

    def forward(self, x):
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


####transformer层
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model_embed, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=d_model_embed)
        self.norm = LayerNorm(d_model_embed)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model_embed, d_ff=feed_forward_hidden, dropout=dropout)


        self.input_sublayer = SublayerConnection(d_model_embed=d_model_embed, dropout=dropout)
        self.output_sublayer = SublayerConnection(d_model_embed=d_model_embed, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        len_x = len(x)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        # print("transformer_inputsublayer", x.size())
        x = torch.sum(x, 1)/len_x
        x = self.output_sublayer(x, self.feed_forward)  ####
        # print("transformer_outputsublayer", x.size())
        # return self.dropout(x)
        return x

####做好嵌入再放进模型
class Transformer(nn.Module):

    def __init__(self, label_embed=None, hidden=768, n_layers=6, attn_heads=12, num_class=9, label_hidden_size=384, atten_size=30, max_sen=20, dropout=0.1):
        super().__init__()
        self.label_embed = label_embed
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        # embedding for BERT, sum of positional, segment, token embeddings

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.num_class = num_class
        self.last = nn.Linear(self.hidden, self.num_class)  ####?

        #########################
        self.label_layer = label_layer(num_class=num_class, embed_dim=hidden, label_hidden_size=label_hidden_size,
                                       max_sen=max_sen, atten_size=atten_size)

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  #####是否修改??
        mask = None
        # embedding the indexed sequence to sequence of vectors

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        # print(x.size())

        # label = self.label_layer(x, self.label_embed)
        # # print(label.size())
        # x = (x + label).cuda()


        x = self.last(x)
        return x