import xlrd
import jieba
import numpy as np
import torch
import os
import random
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
def setup_seed(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(1)


# torch.backends.cudnn.enabled = False
word2vecmodel = Word2Vec.load('./data/YT_word2vec')
Embed_dim = 256

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
    ones = np.ones(Embed_dim, dtype=np.float)  #### 以后 UNK = 1  PAD = 0
    for i in data:
        # temp = []
        temp = np.zeros(Embed_dim, dtype=np.float)
        for j in i:
            try:
                x = word2vecmodel[j]
                # temp.append(x)
                temp = np.add(temp,x)
            except:
                x = ones
                # temp.append(x)
                temp = np.add(temp, x)
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
def get_all_sen_align(embed_data):
    all_sen = []
    for i in range(len(embed_data)):
        embed_data[i] = torch.tensor(embed_data[i], dtype=torch.float32)
        # sen = align(torch.from_numpy(np.array(x_train[i])),32)
        sen = align(embed_data[i], 28)  #####

        all_sen.append(sen)
    return all_sen

data,label = all_data()
# embed_data = data_embed(data)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, stratify=label)
x_train = data_embed(x_train)
x_test = data_embed(x_test)
# x_train = get_all_sen_align(x_train)  # print(all_sen_embed_align)  类型是list  1402个
# x_test = get_all_sen_align(x_test)
# x_train= torch.tensor([item.detach().numpy() for item in x_train])
# x_train=torch.reshape(x_train,[-1,Embed_dim*28])
# x_test= torch.tensor([item.detach().numpy() for item in x_test])
# x_test=torch.reshape(x_test,[-1,Embed_dim*28])

model = GaussianNB()
model.fit(x_train, y_train)
print('训练集准确率：', model.score(x_train, y_train))
print('测试集准确率：', model.score(x_test, y_test))
