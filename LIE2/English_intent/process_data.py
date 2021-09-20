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

# Load pretrained model/tokenizer\
tokenizer = BertTokenizer.from_pretrained('F:\\研二\\LIE\\BERT\\data\\uncased_L-12_H-768_A-12')
BERT_model = BertModel.from_pretrained('F:\\研二\\LIE\\BERT\\data\\uncased_L-12_H-768_A-12')
######


#### 用于分离数据和标签,并去除数据中的停用词
def all_data(trainxlsx):
    # trainxlsx = r'./data/datanews.xlsx'
    book = xlrd.open_workbook(trainxlsx, encoding_override='utf-8')
    trainQues = book.sheet_by_index(0)

    data = []
    label = []
    for row in range(1, trainQues.nrows):
        row_data = trainQues.row_values(row)

        data.append(row_data[0])
        label.append(row_data[1])
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

def get_news_label_embed(max_sen_len):  ###标签关键词的嵌入
    # label = [
    #     "add to playlist,song,album,urban hitd,music",
    #     "book,restaurant,reservation,table,people",
    #     "weather,storm,fog,hot,forecast,warm,rain",
    #     "play,music,open,hear,song,play from",
    #     "rate book,give, points",
    #     "search,find,book,show,look up,look for",
    #     "search screening event,want to,need,find,i,check"
    # ]
    label = [
        "what is mean,what does mean,code,explain,ff,qx,dfw,y,ap,f,h",
        "aircraft,flight,fly,type of, plane,kind of ,airplane, use",
        "aircraft,flight,flight number",
        "airfare,fares,the cost of a flight,fare",
        "airlines,which airlines,fly",
        "airline,flight number",
        "airpot,airpots",
        "how many,list the number of people,total,capacity,seating capacities, how many seats,carried on",
        "cheapest",
        "cities,city,denver,nationair,canadian,washington",
        "how long,distance,how far",
        "flights,from to,leave,fly,flight",
        "flight,fares",
        "the number of flights,flight numbers",
        "the schedule,time",
        "the rental car rates,the cost of limousine,ground transportation,how much,price,taxi,car,fare",
        "is there,transportation,types of ground transportation,kinds of,show me, rental cars,list,taxi,",
        "ground transportation,the cost of,fare",
        "a meal,meals",
        "how many flights,how many cities",
        "restrictions,restriction"
    ]
    label_embed = embed_data(label, max_sen_len)
    return label_embed


