import xlrd
import jieba
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader,TensorDataset,Dataset
from model.LSTM import BP_LSTM

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('.\data\BERT_cased_L-12_H-768_A-12')
BERT_model = BertModel.from_pretrained('.\data\BERT_cased_L-12_H-768_A-12')
######

trainxlsx = r'./data/datatrain.xlsx'
testxlsx = r'./data/datatest.xlsx'

#### 生成停用词列表
stopwords = open('./data/hit_stopwords.txt', encoding='utf-8')
stopwords_list = stopwords.readlines()
stopwords = [x.strip() for x in stopwords_list]    #去掉每行头尾空白

#### 用于分离数据和标签,并去除数据中的停用词
def all_data(trainxlsx):
    # trainxlsx = r'./data/datanews.xlsx'
    book = xlrd.open_workbook(trainxlsx, encoding_override='utf-8')
    trainQues = book.sheet_by_index(0)


    data = []
    label = []
    for row in range(1, trainQues.nrows):
        row_data = trainQues.row_values(row)

        row_data[0] = ''.join([i for i in row_data[0] if not i.isdigit()])

        cut_data = jieba.lcut(row_data[0].strip('\n'))  ## 第一次分词
        ## 去停用词
        final = ''
        for seg in cut_data:
            if seg not in stopwords:
                final += seg
        data.append(final)
        label.append(int(row_data[1]))
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
    label = ["佳能，索尼，手机，尼康，仅售元，相机，三星，单反，评测，液晶，售价，像素，网络，促销",
             "双色球，胜负彩，主场，足球彩票，火线，冷门，开奖，头奖，彩民，第期，任选",
             "均价，开盘，现房，全款，精装，别墅，居在售，在售，楼市，享折，平起，独栋，小户型，平米",
             "港股，评级，震荡，恒指，证券，股市，快讯，反弹，板块，美股，预期，市场，目标价，大盘",
             "家具，家居，装修，地板，卫浴，涂料，家装，瓷砖，装饰，陶瓷",
             "考研，高考，招生，考生，考试，录取，硕士，中考，复习，研究生，高校，志愿，报名，留学，自考，招生简章",
             "基金，期货，商品，行情，分红，震荡，收益，大宗",
             "男子，女子，身亡，司机，村民，妻子，丈夫，民警",
             "减肥，搭配，瘦身，性感，女星，饮食，穿衣，健康，造型",
             "美国，总统，发生，总理，死伤，袭击，武装，媒体，组织，会见",
             "球员，主帅，夺冠，球迷，球队，赛季，首发，替补，冠军，联赛，足球，男篮，决赛",
             "测试，星座，爱情，事业，趣味，职场，运势，心理，本周，吉日，吉时，天秤，水瓶，金牛，白羊，双子，巨蟹，狮子，处女，天蝎，射手，摩羯，双鱼",
             "游戏，网游，玩家，公布，内测，手游",
             "否认，写真，卫视，曝光，导演，经纪人，明星，绯闻，代言"
             ]
    # temp = torch.zeros(1, max_sen_len, Embed_dim)  # torch.from_numpy(model['seg']).unsqueeze(dim=0)
    label_embed = embed_data(label, max_sen_len)
    return label_embed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####一些参数
batch_size = 64
seed = 1
torch.manual_seed(seed)
Embed_dim = 768  ###BERT嵌入
max_sen = 30
dropout = 0.5
hidden_size = 128    ##模型隐藏层维度
bidirectional = True
output_size = 14   ##模型输出层维度
num_class = 14
learning_rate = 0.001     ## 学习率

data, label = all_data(trainxlsx)
test_data, test_label = all_data(testxlsx)
######### train集和val集
x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True, stratify=label)

x_train_embed = embed_data(x_train, max_sen)
x_val_embed = embed_data(x_val, max_sen)
x_test_embed = embed_data(test_data, max_sen)


train_loader = get_dataloder(x_train_embed, y_train, batch_size)
val_loader = get_dataloder(x_val_embed, y_val, batch_size)
test_loader = get_dataloder(x_test_embed, test_label, batch_size)

label_embed = get_news_label_embed(max_sen)


###  构建网络  ###
#
# main_model = BP_LSTM(output_size=output_size,
#                 hidden_size=hidden_size,
#                 embed_dim=Embed_dim,
#                 bidirectional=bidirectional,
#                 dropout=dropout,
#                 sequence_length=max_sen)

# from THUNCnews.atten_lstm import atten_lstm
# main_model = atten_lstm(output_size=output_size,
#                    hidden_size=hidden_size,
#                    embed_dim=Embed_dim,
#                    sequence_length=max_sen,
#                    dropout=dropout)
from model.multi_atten import multi_atten_lstm
main_model = multi_atten_lstm(output_size=output_size,
                              hidden_size=hidden_size,
                              embed_dim=Embed_dim,
                              sequence_length=max_sen,
                              atten_size=32,
                              num_class=num_class,
                              dropout=dropout)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
main_model.to(device)

### 实例化网络
loss_fn = torch.nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)

#### 早停策略
from model.pytorchtools import EarlyStopping
# Train the Model using Early Stopping
patience = 100	# 当验证集损失(或准确率？)在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)	# 关于 EarlyStopping 的代码可先看博客后面的内容
n_epochs = 1000
best_valacc = 0
best_loss = 100
log_dir = 'acc_multi_atten_1.path'

### 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
for epoch in range(1, n_epochs+1):
    ##建立训练数据的dataloader
    main_model.train()
    train_loss = 0
    ##按小批量进行训练
    for batch, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = main_model(data, label_embed)
        # output = main_model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()


    main_model.eval()
    correct_val = 0
    valid_loss = 0
    batch_all = 0
    for batch, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = main_model(data,label_embed)
        # output = main_model(data)
        loss = loss_fn(output, target)
        valid_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct_val += (prediction == target).sum().item()  # .float()
        batch_all += len(data)

    valid_loss = valid_loss/batch_all
    valacc = correct_val/batch_all
    print('valacc:', valacc)
    if valacc > best_valacc:
    # if valid_loss < best_loss:
        # lstm.load_state_dict(torch.load('checkpoint.pt'))  ###保存模型
        best_valacc = valacc
        # best_loss = valid_loss
        state = {'model': main_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log_dir)
    early_stopping(valid_loss, main_model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# print(epoch)

### 测试1

dir = 'checkpoint_multi_atten_1.pt'
checkpoint = torch.load(dir)
main_model.load_state_dict(checkpoint)


main_model.eval()
correct_val = 0
batch_all = 0
for batch, (data, target) in enumerate(test_loader):
    optimizer.zero_grad()
    output = main_model(data, label_embed)
    # output = main_model(data)
    # loss = loss_fn(output, target)
    prediction = torch.argmax(output, 1)
    correct_val += (prediction == target).sum().item()  # .float()
    batch_all += len(data)

valacc = correct_val/batch_all
print('（按loss）testacc: {:.4f}'.format(valacc))
# print(epoch)

###测试2
checkpoint = torch.load(log_dir)
main_model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']

main_model.eval()
correct_val = 0
batch_all = 0
for batch, (data, target) in enumerate(test_loader):
    optimizer.zero_grad()
    output = main_model(data,label_embed)
    # output = main_model(data)
    loss = loss_fn(output, target)
    prediction = torch.argmax(output, 1)
    correct_val += (prediction == target).sum().item()  # .float()
    batch_all += len(data)

valacc = correct_val/batch_all
print('（按acc）valacc:', valacc)







# correct_test = 0
# total_test = 0
# main_model.eval()
# for test, test_label in test_loader:
#     out = main_model(test, label_embed)
#     # out = main_model(test)
#     # 计算准确率
#     prediction = torch.argmax(out, 1)
#     correct_test += (prediction == test_label).sum().item()  # .float()
#     total_test += len(test_label)
# print('testacc: {:.4f}'.format(correct_test / total_test))




##############处理词嵌入文件
# # python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path F:\研一下\chinese_wwm_ext_L-12_H-768_A-12\bert_model.ckpt --bert_config_file F:\研一下\chinese_wwm_ext_L-12_H-768_A-12\bert_config.json --pytorch_dump_path F:\研一下\chinese_wwm_ext_L-12_H-768_A-12\pytorch_model.bin

#
# text = ["我是中国人你是吗"]
#
# x = embed_data(text)
# print(x.size())
# tokenized_text = tokenizer(text, padding=True, truncation=True, max_len=20, return_tensors="pt") #token初始化
# print(tokenized_text)
# print(tokenized_text["input_ids"].shape)
#
# output = model(tokenized_text["input_ids"])
#
# print(output[0].shape)  ##  torch.Size([1, 10, 768])  前后还有两个句子开始结束词
# print(output[1].shape)  ##  torch.Size([1, 768])




