import xlrd
import jieba
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader,TensorDataset,Dataset
from model.LSTM import BP_LSTM

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('F:\研二\LIE\THUNCnews\data\BERT_cased_L-12_H-768_A-12')
BERT_model = BertModel.from_pretrained('F:\研二\LIE\THUNCnews\data\BERT_cased_L-12_H-768_A-12')
######


#### 生成停用词列表
stopwords = open('F:\研二\LIE\YT\data\stopwords.txt', encoding='utf-8')
stopwords_list = stopwords.readlines()
stopwords = [x.strip() for x in stopwords_list]    #去掉每行头尾空白

#### 用于分离数据和标签,并去除数据中的停用词
def all_data(trainxlsx):
    # trainxlsx = r'./data/datanews.xlsx'
    book = xlrd.open_workbook(trainxlsx, encoding_override='utf-8')
    trainQues = book.sheet_by_index(0)

    classdict = {"app": "0", "bus": "1", "calc": "2", "chat": "3", "cinemas": "4", "contacts": "5",
                 "cookbook": "6", "datetime": "7", "epg": "8", "email": "9", "flight": "10",
                 "health": "11", "lottery": "12", "map": "13", "match": "14", "message": "15",
                 "music": "16", "news": "17", "novel": "18", "poetry": "19", "radio": "20",
                 "riddle": "21", "schedule": "22", "stock": "23", "telephone": "24", "train": "25",
                 "translation": "26", "tvchannel": "27", "video": "28", "weather": "29", "website": "30"
                 }

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
        data.append(final)
        label.append(int(classdict[row_data[1]]))
    return data, label

def embed_data(data,max_sen):
    tokenized_text = tokenizer(data, padding='max_length', max_length=max_sen, truncation=True, return_tensors="pt")  # token初始化
    with torch.no_grad():
        output = BERT_model(tokenized_text["input_ids"])
        output = output[0]
    return output

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

def get_news_label_embed(max_sen_len):  ###新闻关键词的嵌入
    label = ["打开，app，软件",
             "坐车，汽车，车",
             "除以，次方，和，平方，加，减，商，乘，根号，立方根，次方根，积，根，差",
             "聊天，你，我，什么",
             "电影，电影院，播放，片子",
             "联系人，电话，号码，手机",
             "怎么做，配方，烧，炒，灼，拌，卤，烤，腊，腌，蒸，煮，煲，羹，汤，面，粥，做法",
             "几点，分钟，日，天，星期，时间",
             "播放，放，有，看，电视台，节目，卫视，央视，中央，CCTV",
             "发，邮件，查看，回复，mail，打开",
             "飞，航班，机票，飞机票，飞机",
             "怎么治，怎么办，治疗",
             "开奖，大乐透，彩票，七乐彩，",
             "路线，怎么走，去，导航，到，在哪里，位置",
             "赛程，结果，预告，比赛",
             "发，信息，短信",
             "播放，唱，歌，一首，听首歌，音乐",
             "新闻，播报，事件，",
             "小说，《》",
             "诗，一首，朗诵，词，后一句，背，念，下一句",
             "调频，收听，广播，电台",
             "灯谜，字谜，脑筋急转弯，谜语",
             "闹钟，叫，提醒，待办",
             "股票，价格，涨跌，市值，股份",
             "打，拨打，电话，呼叫",
             "火车，列车，火车票，动车，高铁",
             "翻译，怎么说，英文",
             "频道，台，CCTV，卫视，央视",
             "电影，演唱会，电视剧",
             "天气，预报，几度",
             "网，网站，网页"
             ]
    # temp = torch.zeros(1, max_sen_len, Embed_dim)  # torch.from_numpy(model['seg']).unsqueeze(dim=0)
    label_embed = embed_data(label, max_sen_len)
    return label_embed


