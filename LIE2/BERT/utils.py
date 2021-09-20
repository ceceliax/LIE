# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from transformers import BertTokenizer, BertModel
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('.\data\BERT_cased_L-12_H-768_A-12')
BERT_model = BertModel.from_pretrained('.\data\BERT_cased_L-12_H-768_A-12')

# tokenizer = BertTokenizer.from_pretrained('.\\data\\uncased_L-12_H-768_A-12')
# BERT_model = BertModel.from_pretrained('.\\data\\uncased_L-12_H-768_A-12')


# train_path = './data/traindata.txt'
# test_path = './data/testdata.txt'
# dev_path = './data/dev.txt'


# train_path = './data/snips_train.txt'
# test_path = './data/snips_test.txt'
# dev_path = './data/snips_dev.txt'

# train_path = './data/atis_train.txt'
# test_path = './data/atis_test.txt'
# dev_path = './data/atis_dev.txt'

train_path = './data/chinese_intent_train.txt'
test_path = './data/chinese_intent_test.txt'
dev_path = './data/chinese_intent_dev.txt'

# train_path = './data/YT_train.txt'
# test_path = './data/YT_test.txt'
# dev_path = './data/YT_dev.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dataset(pad_size):

    def load_dataset(path, pad_size):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    dev = load_dataset(dev_path, pad_size)

    test = load_dataset(test_path, pad_size)
    train = load_dataset(train_path, pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter


# def get_time_dif(start_time):
#     """获取已使用时间"""
#     end_time = time.time()
#     time_dif = end_time - start_time
#     return timedelta(seconds=int(round(time_dif)))




def embed_data(data,max_sen):
    tokenized_text = tokenizer(data, padding='max_length', max_length=max_sen, truncation=True, return_tensors="pt")  # token初始化
    with torch.no_grad():
        output = BERT_model(tokenized_text["input_ids"])
        output = output[0]
    return output

def get_news_label_embed(max_sen_len):  ###新闻关键词的嵌入
    # label = [
    #     "add to playlist,song,album,urban hitd,music",
    #     "book,restaurant,reservation,table,people",
    #     "weather,storm,fog,hot,forecast,warm,rain",
    #     "play,music,open,hear,song,play from",
    #     "rate book,give, points",
    #     "search,find,book,show,look up,look for",
    #     "search screening event,want to,need,find,i,check"
    # ]

    label = ["打开，软件",
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
    # label = [ "查快递，查件，快递在哪，查包裹",
    #           "派件，派件员，收件，收货，派送，配送，收派件",
    #           "标准，注意事项，加盟，什么意思",
    #           "转运，中转站，转同行，未更新，延误，退回，查验，扣货，范围，时效",
    #           "寄快递，寄件，快递单，下单，邮寄",
    #           "投诉，太慢了，态度差，骚扰，错，打不通",
    #           "签收，代收，自提，取货码，取件",
    #           "仓租费，退件费，附加费，收费，税费，运费，费用",
    #           "转运中心，站点电话，网点地址，代办点，网点信息"
    #          ]
    # label = [
    #     "what is mean,what does mean,code,explain,ff,qx,dfw,y,ap,f,h",
    #     "aircraft,flight,fly,type of, plane,kind of ,airplane, use",
    #     "aircraft,flight,flight number",
    #     "airfare,fares,the cost of a flight,fare",
    #     "airlines,which airlines,fly",
    #     "airline,flight number",
    #     "airpot,airpots",
    #     "how many,list the number of people,total,capacity,seating capacities, how many seats,carried on",
    #     "cheapest",
    #     "cities,city,denver,nationair,canadian,washington",
    #     "how long,distance,how far",
    #     "flights,from to,leave,fly,flight",
    #     "flight,fares",
    #     "the number of flights,flight numbers",
    #     "the schedule,time",
    #     "the rental car rates,the cost of limousine,ground transportation,how much,price,taxi,car,fare",
    #     "is there,transportation,types of ground transportation,kinds of,show me, rental cars,list,taxi,",
    #     "ground transportation,the cost of,fare",
    #     "a meal,meals",
    #     "how many flights,how many cities",
    #     "restrictions,restriction"
    # ]
    # temp = torch.zeros(1, max_sen_len, Embed_dim)  # torch.from_numpy(model['seg']).unsqueeze(dim=0)
    label_embed = embed_data(label, max_sen_len)
    # print("label_embed",label_embed.size())
    return label_embed