import torch.nn as nn
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(0)

class BP_LSTM(nn.Module):

    def __init__(self,  output_size, hidden_size, embed_dim, bidirectional,
                 dropout,sequence_length):
        super(BP_LSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional
                            )

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        # self.label = nn.Linear(hidden_size * self.layer_size * self.sequence_length, 9)#128
        self.label = nn.Linear(hidden_size * self.layer_size, self.output_size)
        # self.out1 = nn.Linear(128,64)
        # self.out2 = nn.Linear(64,30)
        # self.out3 = nn.Linear(30,self.output_size)
    def init_hidden(self,batch_size):
        return (torch.zeros(self.layer_size,batch_size,self.hidden_size,dtype=torch.float32).to(device),
                torch.zeros(self.layer_size,batch_size,self.hidden_size,dtype=torch.float32).to(device))


    def forward(self, x):
        batch_size = x.size(0)  ### 16
        hidden = self.init_hidden(batch_size)
        # x的size:batch_size*sequence_length*embed_dim
        # x = x.permute(1, 0, 2)
        # h_0 = Variable(torch.zeros(self.layer_size, batch_size, self.hidden_size))
        # c_0 = Variable(torch.zeros(self.layer_size, batch_size, self.hidden_size))
        # h_0包含的是句子的最后一个单词的隐藏状态，c_0包含的是句子的最后一个单词的细胞状态，
        # 所以它们都与句子的长度seq_length无关。

        lstm_output, _ = self.lstm(x, hidden)
        # lstm_output, _ = self.lstm(x, (h_0,c_0))
        # lstm_output的shape==(batch_size,seq_length,layer_size×hidden_size)
        lstm_output = torch.sum(lstm_output, 1)  ###将seq_length这一维相加，形成句子向量
        lstm_output = lstm_output.reshape(batch_size,-1)
        logits = self.label(lstm_output)
        return logits

