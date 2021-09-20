import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, class_num, embed_dim, label,dropout):
        super(CNN_Text, self).__init__()

        Ci = 1
        self.label = label
        self.class_num = class_num
        # self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num * 2, class_num)


    def sen_attention_label(self,x, label):
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
        finalx = finalx.transpose(0, 1).cuda()
        # lstm_output = torch.sum(lstm_output, 1)
        # out = torch.cat((sen_attn_output, finalx), dim = 1)  #横着拼  (batch_size, hidden_size*layer_size*2)
        out = torch.cat((x, finalx), dim=1).cuda()
        # lstm_output = torch.sum(lstm_output, 1)
        # output = torch.cat((lstm_output, out), dim = 1)
        return out


    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        label = self.label.unsqueeze(1)  # (N, Ci, W, D)
        label = [F.relu(conv(label)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        label = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in label]  # [(N, Co), ...]*len(Ks)
        label = torch.cat(label, 1)

        x = self.sen_attention_label(x, label)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
