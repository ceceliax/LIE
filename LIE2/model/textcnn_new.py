import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, class_num, embed_dim,  max_sen_len,atten_size,label, dropout):
        super(CNN_Text, self).__init__()

        Ci = 1
        self.label = label
        self.class_num = class_num
        self.embed_dim = embed_dim
        self.hidden_size = kernel_num*len(kernel_sizes)//2
        self.sen_len = max_sen_len
        self.attention_size = atten_size
        # self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, self.embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)

        self.w_omega = Variable(torch.zeros(self.hidden_size * 2, self.attention_size).cuda())
        # （30）
        self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, 1, batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (batch_size,squence_length,  hidden_size*layer_size)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * 2]).cuda()
        # print(output_reshape.size()) = ( batch_size*squence_length, hidden_size*layer_size)
        # tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega)).cuda()
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        # 张量相乘
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1])).cuda()
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sen_len]).cuda()
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1]).cuda()
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sen_len, 1]).cuda()
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        # ###这边的sum是把sequence_length这一维加起来了，所以最后self.label层不需要*sequence_label了
        attn_output = torch.sum(lstm_output * alphas_reshape, 1).cuda()
        # lstm_output = torch.sum(lstm_output,1)
        # out = torch.cat((attn_output,lstm_output),dim=1)
        return attn_output
    def sen_attention_label(self, x, label):
        # x.size():(batch_size, len(kernel_sizes) * kernel_num)  query
        # label:(num_class, len(kernel_sizes) * kernel_num)  Xi
        ### 计算方式点积
        label = label.transpose(0, 1).cuda()
        # print('label.size():',label.size())
        m = torch.tanh(torch.mm(x, label)).cuda()
        exps = torch.exp(m).cuda()
        a = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1]).cuda()  ###算权重
        a_reshape = torch.Tensor.reshape(a, [self.class_num, -1]).cuda()
        # print('a_reshape',a_reshape.size())
        # label_attn_output = label_attn_output.transpose(0, 1)
        # print('label_attn_output.size():', label_attn_output.size())
        finalx = torch.mm(label, a_reshape).cuda()
        finalx = finalx.transpose(0,1).cuda()

        # finalx = torch.mm(x, finalx)
        # print('finalx',finalx.size())

        # lstm_output = torch.sum(lstm_output, 1)
        # out = torch.cat((sen_attn_output, finalx), dim = 1)  #横着拼  (batch_size, hidden_size*layer_size*2)
        # out = torch.cat((x, finalx), dim=1)
        out = (x+finalx).cuda()
        # print('out', out.size())
        # lstm_output = torch.sum(lstm_output, 1)
        # output = torch.cat((lstm_output, out), dim = 1)
        return out


    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        s, b, f = self.label.size()
        h_0 = Variable(torch.zeros(2, s, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, s, self.hidden_size).cuda())
        label, (final_hidden_state, final_cell_state) = self.lstm(self.label, (h_0, c_0))
        # label = torch.sum(label, 1)
        label = self.attention_net(label)

        x = self.sen_attention_label(x, label)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
