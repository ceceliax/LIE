import xlrd
import jieba
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
# from BERT.utils import get_news_label_embed
import random
import time
from datetime import timedelta
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('.\data\BERT_cased_L-12_H-768_A-12')
BERT_model = BertModel.from_pretrained('.\data\BERT_cased_L-12_H-768_A-12')


# tokenizer = BertTokenizer.from_pretrained('.\\data\\uncased_L-12_H-768_A-12')
# BERT_model = BertModel.from_pretrained('.\\data\\uncased_L-12_H-768_A-12')
######
#### 生成停用词列表
# stopwords = open('./data/hit_stopwords.txt', encoding='utf-8')
# stopwords_list = stopwords.readlines()
# stopwords = [x.strip() for x in stopwords_list]    #去掉每行头尾空白

def load_dataset(path, max_sen):
    data_list = []
    label_list = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            content = content.strip('\n')
            # cut_data = jieba.lcut(content.strip('\n'))  ## 第一次分词
            # ## 去停用词
            # final = ''
            # for seg in cut_data:
            #     if seg not in stopwords:
            #         final += seg
            tokenized_text = tokenizer(content, padding='max_length', max_length=max_sen, truncation=True,
                                       return_tensors="pt")  # token初始化
            with torch.no_grad():
                output = BERT_model(tokenized_text["input_ids"])
                output = output[0]
            data_list.append(output)
            label_list.append(int(label))
    return data_list, label_list


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

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))