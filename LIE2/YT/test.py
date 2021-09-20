import torch
import xlrd
import jieba
import torch.nn.functional as F
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

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
        # final = jieba.lcut(final.strip('\n'))  ##去停用词后重新分词
        ##去除列表中的空格字符串
        # for i in final:
        #     if i == ' ':
        #         final.remove(' ')
        data.append(final)
        label.append(int(classdict[row_data[1]]))
    return data, label

data,label = all_data()

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=44, stratify=label)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=44, stratify=y_train)

for i in range(len(x_train)):
    w = open('./data/YT_train.txt', 'a', encoding='utf-8')
    w.write(x_train[i] + '\t' + str(y_train[i]) + '\n')
    w.close()
for i in range(len(x_test)):
    w = open('./data/YT_test.txt', 'a', encoding='utf-8')
    w.write(x_test[i] + '\t' + str(y_test[i]) + '\n')
    w.close()
for i in range(len(x_val)):
    w = open('./data/YT_dev.txt', 'a', encoding='utf-8')
    w.write(x_val[i] + '\t' + str(y_val[i]) + '\n')
    w.close()