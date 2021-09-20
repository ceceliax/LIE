
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


####主
class multi_atten_lstm(torch.nn.Module):
    def __init__(self, output_size, hidden_size, embed_dim, sequence_length, num_class,
                  atten_size, label_embed, label_max_sen, dropout):
        super(multi_atten_lstm, self).__init__()

        self.label_embed = label_embed

        self.output_size = output_size
        self.hidden_size = hidden_size
        #对应特征维度
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_class = num_class
        #对应时间步长
        self.sequence_length = sequence_length
        self.bidirectional = True
        #1层lstm
        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,self.hidden_size,self.layer_size, dropout=self.dropout,
                            batch_first=True,
                            bidirectional= self.bidirectional
                            )#(squence_length, batch_size, hidden_size*layer_size)
        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size
        # self.layer_size = self.layer_size
        self.attention_size = atten_size

        self.label_layer = label_layer(num_class=self.num_class, embed_dim=embed_dim, label_hidden_size=hidden_size,
                                       max_sen=label_max_sen, atten_size=atten_size)

        self.last = nn.Linear(hidden_size * self.layer_size, output_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.layer_size,batch_size,self.hidden_size,dtype=torch.float32).cuda(),
                torch.zeros(self.layer_size,batch_size,self.hidden_size,dtype=torch.float32).cuda())



    def forward(self, input):
        batch_size = input.size(0)
        input_hidden = self.init_hidden(batch_size)

        lstm_output, _ = self.lstm(input, input_hidden)
        lstm_output = torch.sum(lstm_output, 1)
        label = self.label_layer(lstm_output,self.label_embed)

        # ###形成句子向量
        # lstm_output = torch.sum(lstm_output, 1)
        # label_lstm_out = torch.sum(label_lstm_out, 1)

        out = (lstm_output+label).cuda()

        logits = self.last(out)
        return logits