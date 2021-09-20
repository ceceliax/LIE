# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
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


class Model(nn.Module):

    def __init__(self, bert_path, hidden_size, num_classes, label_embed, embed_dim, label_hidden_size, max_sen, atten_size):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.label_embed = label_embed
        self.label_hidden_size = label_hidden_size
        self.dropout = nn.Dropout(0.1)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_classes)
        self.label_layer = label_layer(num_class=num_classes,embed_dim=embed_dim,label_hidden_size=label_hidden_size,max_sen=max_sen,atten_size=atten_size)


    def forward(self, x):
        context = x[0].cuda()  # 输入的句子
        mask = x[2].cuda()  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask)


        label = self.label_layer(pooled, self.label_embed)
        # self.pooled_in = pooled.detach()
        pooled = (pooled+label).cuda()
        # self.pooled_out = pooled.detach()
        # pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out
