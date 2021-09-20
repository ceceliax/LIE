import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from BERT.bert import Model
from BERT.utils import build_dataset, build_iterator, get_news_label_embed
import pandas as pd

num_class = 31
batch_size = 1

pad_size = 32                                              # 每句话处理成的长度(短填长切)
max_sen_len = 32
learning_rate = 5e-6

hidden_size = 768 ###embed_dim
Embed_dim = 768



label_embed = get_news_label_embed(max_sen_len).cuda()
bert_path = '.\data\BERT_cased_L-12_H-768_A-12'

main_model = Model(bert_path=bert_path,
              hidden_size=hidden_size,
              num_classes=num_class,
              label_embed=label_embed,
              embed_dim=Embed_dim,
              atten_size=30,
              label_hidden_size=384,
              max_sen=max_sen_len
              ).cuda()

train_data, dev_data, test_data = build_dataset(pad_size)
test_iter = build_iterator(test_data, batch_size)
### 测试1
dir = './data/bert.ckpt'

checkpoint = torch.load(dir)
main_model.load_state_dict(checkpoint)
#
# for name in main_model.state_dict():
#     print(name)

# print(main_model.state_dict()['label_layer.lstm.weight_ih_l0)'])

main_model.eval()
loss_total = 0
predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)


with torch.no_grad():
    for texts, labels in test_iter:
        outputs = main_model(texts)
        loss = F.cross_entropy(outputs, labels)
        loss_total += loss
        labels = labels.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)
# print(main_model.state_dict()['label_layer.lstm.weight_ih_l0'])
acc = metrics.accuracy_score(labels_all, predict_all)
recall = metrics.recall_score(labels_all, predict_all, average='weighted')
f1_score = metrics.f1_score(labels_all, predict_all, average='weighted')
print(acc)
print(recall)
print(f1_score)


pool_in = main_model.pooled_in.cpu()

a = main_model.label_layer.sen_attention_label.a.cpu().numpy()

label = [
    "add, playlist,song,album,urban hitd,music",
    "book,restaurant,reservation,table,people",
    "weather,storm,fog,hot,forecast,warm,rain",
    "play,music,open,hear,turn on,song,",
    "rate book,give, points",
    "search creative work,find,look up,look for",
    "search screening event,want to,need,find,i,check"
]
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



label2 = [
    "the rental car rates,the cost of limousine,ground transportation,how much,price,taxi,car",
    "mean,what is,code,ff,qx,dfw,y,ap,f,h",
    "how long,distance,how far",
    "restrictions,restriction",
    "is there,types of ground transportation,kinds of,show me, rental cars,list,taxi,",
    "aircraft,flight,fly,planes,airplane",
    "how many,list the number",
    "airpot,airpots",
    "cheapest",
    "cities,city,denver,nationair,canadian,washington",
    "seating capacities,how many, how many seats,carried on",
    "a meal,meals",
    "the schedule,time",
    "airlines,which,airline",
    "flights,from to,leave,fly",
    "the number of flights,flight numbers",
    "airfare,fares,the cost of a flight,fare"
]


