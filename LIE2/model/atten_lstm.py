import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
class atten_lstm(torch.nn.Module):
    def __init__(self, output_size, hidden_size, embed_dim, sequence_length,attention_size, dropout):
        super(atten_lstm, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        #对应特征维度
        self.embed_dim = embed_dim
        self.dropout = dropout
        #对应时间步长
        self.sequence_length = sequence_length
        self.bidirectional = True
        #1层lstm
        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            batch_first=True,
                            bidirectional= self.bidirectional
                            )#(squence_length, batch_size, hidden_size*layer_size)
        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size
        # self.layer_size = self.layer_size
        self.attention_size = attention_size
        #（4，30）
        self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
        #（30）
        self.u_omega = Variable(torch.zeros(self.attention_size))
        #将隐层输入全连接
        # self.label = nn.Linear(hidden_size * self.layer_size*self.sequence_length, output_size)
        self.label = nn.Linear(hidden_size * self.layer_size * 2 , output_size)  ##*2是因为做了残差拼接

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (batch_size,squence_length,  hidden_size*layer_size)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = ( batch_size*squence_length, hidden_size*layer_size)
        # tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        # 张量相乘
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        # ###这边的sum是把sequence_length这一维加起来了，所以最后self.label层不需要*sequence_label了
        attn_output = torch.sum(lstm_output * alphas_reshape, 1)
        lstm_output = torch.sum(lstm_output,1)
        out = torch.cat((attn_output,lstm_output),dim=1)

        return out

    # def attention_net(self, query, key, value, mask=None):
    #     d_k = query.size(-1)
    #     scores = torch.matmul(query, key.transpose(-2, -1)) \
    #              / math.sqrt(d_k)
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #     p_attn = F.softmax(scores, dim=-1)
    #     print(p_attn.size())
    #     return torch.matmul(p_attn, value)


    def forward(self, input):
        # input = self.lookup_table(input_sentences)
        batch_size = input.size(0)
        # input = input.permute(1, 0, 2)
        # print('input.size():',input.size())
        s, b, f = input.size()
        h_0 = Variable(torch.zeros(self.layer_size, s, self.hidden_size))
        c_0 = Variable(torch.zeros(self.layer_size, s, self.hidden_size))
        # print('input.size(),h_0.size(),c_0.size()', input.size(), h_0.size(), c_0.size())
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        # attn_output = self.attention_net(lstm_output,lstm_output,lstm_output)
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits
