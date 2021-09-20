import torch
from English_intent.process_data import all_data, embed_data, get_dataloder,get_news_label_embed
import os
import random
import numpy as np
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainxlsx = r'./snips/snips_train.xlsx'
testxlsx = r'./snips/snips_test.xlsx'
devxlsx = r'./snips/snips_dev.xlsx'
print(torch.cuda.is_available())

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
####一些参数
batch_size = 64
learning_rate = 5e-3 ## 学习率


seed = 1
seed_torch(seed)
Embed_dim = 768  ###BERT嵌入
max_sen_sen = 30
max_sen = 20  ##label嵌入长度
dropout = 0.5
hidden_size = 128    ##模型隐藏层维度
bidirectional = True
# output_size = 31   ##模型输出层维度
num_class = 21

traindata, trainlabel = all_data(trainxlsx)
testdata, testlabel = all_data(testxlsx)
devdata, devlabel = all_data(devxlsx)

x_train_embed = embed_data(traindata, max_sen_sen)
x_val_embed = embed_data(devdata, max_sen_sen)
x_test_embed = embed_data(testdata, max_sen_sen)

train_loader = get_dataloder(x_train_embed, trainlabel, batch_size)
val_loader = get_dataloder(x_val_embed, devlabel, batch_size)
test_loader = get_dataloder(x_test_embed, testlabel, batch_size)

label_embed = get_news_label_embed(max_sen).cuda()

###  构建网络  ###
from model.LSTM import BP_LSTM
main_model = BP_LSTM(output_size=num_class,
                hidden_size=hidden_size,
                embed_dim=Embed_dim,
                bidirectional=bidirectional,
                dropout=dropout,
                sequence_length=max_sen).cuda()


# from model.multi_atten import multi_atten_lstm
# main_model = multi_atten_lstm(output_size=num_class,
#                               hidden_size=hidden_size,
#                               embed_dim=Embed_dim,
#                               sequence_length=max_sen,
#                               atten_size=32,
#                               num_class=num_class,
#                               label_embed=label_embed,
#                               label_max_sen=max_sen,
#                               dropout=dropout).cuda()

# from model.textCNN import CNN_Text
# main_model = CNN_Text(kernel_num=100,
#                       kernel_sizes=[3,4,5],
#                       class_num=num_class,
#                       embed_dim=Embed_dim,
#                       dropout=dropout).cuda()


# from model.textcnn_new import CNN_Text
# main_model = CNN_Text(kernel_num=100,
#                  kernel_sizes=[3, 4, 5],
#                  class_num=num_class,
#                  embed_dim=Embed_dim,
#                  max_sen_len=max_sen,
#                  atten_size=32,
#                  label=label_embed,
#                  dropout=dropout).cuda()


### 实例化网络
loss_fn = torch.nn.CrossEntropyLoss().cuda() #交叉熵损失函数
optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)
lr_list = []
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.9)


#### 早停策略

n_epochs = 1000
best_valacc = 0
best_loss = 100
dir = 'checkpoint_multi_atten_1.pt'
log_dir = 'acc_multi_atten_1.path'

require_improvement = 1000
total_batch = 0  # 记录进行到多少batch
dev_best_acc = 0
dev_best_loss = float('inf')
last_improve = 0  # 记录上次验证集loss下降的batch数
flag = False  # 记录是否很久没有效果提升


def evaluate(main_model, dev_dataloader):
    main_model.eval()
    correct_val = 0
    valid_loss = 0
    batch_all = 0
    for batch, (data, target) in enumerate(dev_dataloader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        # output = main_model(data,label_embed)
        output = main_model(data)
        loss = loss_fn(output, target)
        valid_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct_val += (prediction == target).sum().item()  # .float()
        batch_all += len(data)
    valacc = correct_val / batch_all
    return valacc,valid_loss/len(data)



### 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
for epoch in range( n_epochs):

    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    print('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
    ##建立训练数据的dataloader
    main_model.train()
    train_loss = 0
    ##按小批量进行训练
    for batch, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        # output = main_model(data, label_embed)
        output = main_model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if total_batch % 10 == 0:
            # 每多少轮输出在训练集和验证集上的效果
            true = target.data.cpu()
            predic = torch.max(output.data, 1)[1].cpu()

            train_acc = metrics.accuracy_score(true, predic)
            dev_acc, dev_loss = evaluate(main_model, val_loader)
            test_, test_loss = evaluate(main_model, test_loader)
            if dev_acc > dev_best_acc:
                dev_best_acc = dev_acc
                torch.save(main_model.state_dict(), log_dir)
            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(main_model.state_dict(), dir)
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''

            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},Test Acc: {5:>6.2%},  {6} '
            print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, test_,improve))
            main_model.train()
        total_batch += 1
        if total_batch - last_improve > require_improvement:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break
            #         # 验证集loss超过1000batch没下降，结束训练

    if flag:
        break


### 测试1

checkpoint = torch.load(dir)
main_model.load_state_dict(checkpoint)
main_model.eval()
correct_val = 0
batch_all = 0
for batch, (data, target) in enumerate(test_loader):
    optimizer.zero_grad()
    data = data.cuda()
    target = target.cuda()
    # output = main_model(data, label_embed)
    output = main_model(data)
    # loss = loss_fn(output, target)
    prediction = torch.argmax(output, 1)
    correct_val += (prediction == target).sum().item()  # .float()
    batch_all += len(data)

valacc = correct_val/batch_all
print('（按loss）testacc: {:.4f}'.format(valacc))
# print(epoch)

# ###测试2
checkpoint = torch.load(log_dir)
main_model.load_state_dict(checkpoint)
main_model.eval()
correct_val = 0
batch_all = 0
for batch, (data, target) in enumerate(test_loader):
    optimizer.zero_grad()
    data = data.cuda()
    target = target.cuda()
    # output = main_model(data, label_embed)
    output = main_model(data)
    # loss = loss_fn(output, target)
    prediction = torch.argmax(output, 1)
    correct_val += (prediction == target).sum().item()  # .float()
    batch_all += len(data)

valacc = correct_val/batch_all
print('（按acc）valacc:', valacc)