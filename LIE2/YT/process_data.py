import torch
import xlrd
import jieba
import torch.nn.functional as F
import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader,TensorDataset,Dataset

word2vecmodel = Word2Vec.load('./data/YT.model')
Embed_dim = word2vecmodel.wv.vector_size

# Embed_dim = 256
#### 生成停用词列表
stopwords = open('./data/stopwords.txt', encoding='utf-8')
stopwords_list = stopwords.readlines()
stopwords = [x.strip() for x in stopwords_list]    #去掉每行头尾空白

#### 用于分离数据和标签,并去除数据中的停用词
def all_data():
    trainxlsx = r'./data/clear_new_intent.xlsx'
    book = xlrd.open_workbook(trainxlsx, encoding_override='utf-8')
    trainQues = book.sheet_by_index(0)

    classdict = {"查件": "0", "收派件": "1", "业务规则": "2", "中转运输": "3", "寄件": "4", "抱怨": "5",
                 "收件": "6", "费用": "7", "网点信息": "8"}
    data = []
    label = []
    for row in range(1, trainQues.nrows):
        row_data = trainQues.row_values(row)
        cut_data = jieba.lcut(row_data[0].strip('\n'))  ## 第一次分词
        ## 去停用词
        final = ''
        for seg in cut_data:
            if seg not in stopwords:
                final += seg
        final = jieba.lcut(final.strip('\n'))  ##去停用词后重新分词
        ##去除列表中的空格字符串
        for i in final:
            if i == ' ':
                final.remove(' ')
        data.append(final)
        label.append(int(classdict[row_data[1]]))
    return data, label

def data_embed(data):
    embed_data = []
    ones = np.ones(Embed_dim, dtype=np.float)
    for i in data:
        temp = []
        for j in i:
            try:
                x = word2vecmodel[j]
                temp.append(x)
            except:
                x = ones
                temp.append(x)
        embed_data.append(temp)
    return embed_data

def align(sen, max_sen_len):
    twolen = sen.size()[0]
    if twolen < max_sen_len:  # 超过最大长度，则截取
        zeros = torch.zeros(max_sen_len - twolen, Embed_dim)
        sen = torch.cat((sen, zeros), dim=0)
        # sen = sen.reshape(1, max_sen_len*Embed_dim)
    else:
        sen = sen[:max_sen_len, :]
        # sen = sen.reshape(1, max_sen_len * Embed_dim)
    return sen

def get_all_sen_align(embed_data,max_sen_len):
    all_sen = []
    for i in range(len(embed_data)):
        embed_data[i] = torch.tensor(embed_data[i], dtype=torch.float32)
        # sen = align(torch.from_numpy(np.array(x_train[i])),32)
        sen = align(embed_data[i], max_sen_len)  #####

        all_sen.append(sen)
    return all_sen

####label_embed  标签嵌入（只做词嵌入）
def word_2_vec(psglist):   ##将单词转换为向量，词向量  #和词性向量
    temp = torch.zeros(1,Embed_dim, dtype=torch.float)
    ones = np.ones(Embed_dim, dtype=np.float)
    for item in psglist:
        try:
            wordvec = word2vecmodel[item]  # 获得当前词的词向量
        except KeyError:
            wordvec = ones  # 若没有该词，则为0，下同，获得当前词的词向量
        # temp = torch.cat((temp, torch.from_numpy(wordvec).float()))
        temp = torch.cat((temp, torch.from_numpy(wordvec).float().unsqueeze(dim=0)), dim=0)
    out = temp[1:, :]
    return out

def get_one2many_YT_label_embed(max_sen_len):  ###圆通关键词的嵌入

    label = {"查件": "查快递，查件，快递在哪，查包裹",
             "收派件": "派件，派件员，收件，收货，派送，配送，收派件",
             "业务规则": "标准，注意事项，加盟，什么意思",
             "中转运输": "转运，中转站，转同行，未更新，延误，退回，查验，扣货，范围，时效",
             "寄件": "寄快递，寄件，快递单，下单，邮寄",
             "抱怨": "投诉，太慢了，态度差，骚扰，错，打不通",
             "取件": "签收，代收，自提，取货码，取件",
             "费用": "仓租费，退件费，附加费，收费，税费，运费，费用",
             "网点信息": "转运中心，站点电话，网点地址，代办点，网点信息"
             }

    # label = {"查件": "快递，查询，一下，查件，单号，手机号，物流，查不到，运单，信息，我查，查下，",
    #          "收派件": "派送，快递，取件，收件，上门，联系，派件，配送，到达",
    #          "业务规则": "标准，加盟，快递，理赔，圆通，限制，收费，批价，注意事项，意思，请问，重量，代收，保价",
    #          "中转运输": "快递，地址，包裹，修改，更新，更改，清关，显示，物流，海关，签收，快件，转运，仓库，退回，收货，写错，信息，入库",
    #          "寄件": "下单，包装，快递，寄件，填写，收件人，信息，订单，国际，春节，添加，我要",
    #          "抱怨": "投诉，快递，太慢，我要，不通，电话，业务员，垃圾，太差",
    #          "取件": "签收，收到，显示，包裹，快递，发现，没有，少件，拒收，失败",
    #          "费用": "收费，运费，付款，保价，订单，包裹，需要，理赔，快递，开发票，租费，贷款，价格，合并，发票，付钱，加盟费",
    #          "网点信息": "网点，电话，附近，范围，地址，一下，驿站，派送，快递，圆通，服务，查询，菜鸟，联系方式，营业部"
    #          }
    temp = torch.zeros(1, max_sen_len, Embed_dim)  # torch.from_numpy(model['seg']).unsqueeze(dim=0)
    for key, value in label.items():
        label_psg_list = jieba.lcut(value)
        ## 去停用词
        final = ''
        for seg in label_psg_list:
            if seg not in stopwords:
                final += seg
        final = jieba.lcut(final.strip('\n'))  ##去停用词后重新分词

        label_embed = word_2_vec(final)
        sen_len = label_embed.size()[0]
        if sen_len < max_sen_len:
            zeros = torch.zeros(max_sen_len - sen_len, Embed_dim)
            label_embed = torch.cat((label_embed, zeros), dim=0)
        else:
            label_embed = label_embed[:max_sen_len, :]
        temp = torch.cat((temp, label_embed.unsqueeze(dim=0)), dim=0)
    aligen = temp[1:, :, :]
    return aligen

####data_loader
def get_dataloder(train_embed_align, y_train, batch_size):
    # train_embed_align = np.array(train_embed_align)
    train_embed_align= [t.numpy() for t in train_embed_align]  ## 转成numoy格式，由于torch有多维时不能直接转
    train_embed_align = np.array(train_embed_align)
    train_embed_align = torch.Tensor(train_embed_align)
    y_train = np.array(y_train)
    y_train = torch.LongTensor(y_train)
    deal_dataset = TensorDataset(train_embed_align, y_train)
    data_loader = DataLoader(dataset=deal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return data_loader