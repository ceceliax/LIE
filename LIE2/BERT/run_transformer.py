import xlrd
import jieba
import torch
import time
from sklearn import metrics
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
from BERT.utils import get_news_label_embed
from BERT.data_for_transfomer import load_dataset,get_dataloder,get_time_dif
label_embed = get_news_label_embed(20)
label_embed = label_embed.cuda()

train_path = './data/chinese_intent_train.txt'
test_path = './data/chinese_intent_test.txt'
dev_path = './data/chinese_intent_dev.txt'

# train_path = './data/atis_train.txt'
# test_path = './data/atis_test.txt'
# dev_path = './data/atis_dev.txt'

# train_path = './data/YT_train.txt'
# test_path = './data/YT_test.txt'
# dev_path = './data/YT_dev.txt'
seed = 32

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_torch(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_sen = 32
batch_size = 128
LR = 5e-6

train, train_label = load_dataset(train_path, max_sen)
dev, dev_label = load_dataset(dev_path, max_sen)
test, test_label = load_dataset(test_path, max_sen)
# train, test, train_label, test_label = train_test_split(test, test_label, test_size=0.2, random_state=1, shuffle=True, stratify=test_label)


train_dataloader = get_dataloder(train, train_label, batch_size)
dev_dataloader = get_dataloder(dev, dev_label, batch_size)
test_dataloader = get_dataloder(test, test_label, batch_size)

from BERT.transformer import Transformer
main_model = Transformer(num_class=31, label_embed=label_embed).cuda()

dir = 'checkpoint_multi_atten_1.pt'
log_dir = 'acc_multi_atten_1.path'

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样
### 实例化网络
loss_fn = torch.nn.CrossEntropyLoss() #交叉熵损失函数
from transformers.optimization import AdamW
start_time = time.time()
lr_list = []
optimizer = torch.optim.AdamW(main_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.9)

#### 早停策略
n_epochs = 1000
best_valacc = 0
best_loss = 100

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
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        # output = main_model(data,label_embed)
        output = main_model(data)
        loss = loss_fn(output, target)
        valid_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct_val += (prediction == target).sum().item()  # .float()
        batch_all += len(data)
    valacc = correct_val / batch_all
    return valacc,valid_loss/batch_all
#
#
#
### 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
for epoch in range( n_epochs):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    print('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
    ##建立训练数据的dataloader
    main_model.train()
    train_loss = 0
    ##按小批量进行训练
    for batch, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        target = target.to(device)
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
            dev_acc, dev_loss = evaluate(main_model, dev_dataloader)
            test_, test_loss = evaluate(main_model, test_dataloader)
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
            time_dif = get_time_dif(start_time)
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},Test Acc: {5:>6.2%}, Time: {6} {7}'
            print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, test_,improve, time_dif))
            main_model.train()
        total_batch += 1
        if total_batch - last_improve > require_improvement:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break
            #         # 验证集loss超过1000batch没下降，结束训练

    if flag:
        break



import torch.nn.functional as F
def evaluate2(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.cuda()
            labels = labels.cuda()
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc
### 测试1

checkpoint = torch.load(dir)
main_model.load_state_dict(checkpoint)
main_model.eval()
correct_val = 0
batch_all = 0
with torch.no_grad():
    # test_, test_loss = evaluate(main_model, test_dataloader)
    # acc = evaluate2(main_model, test_dataloader)
    # print(test_,' ',acc)
    for batch, (data, target) in enumerate(test_dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
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
with torch.no_grad():
    for batch, (data, target) in enumerate(test_dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        # output = main_model(data, label_embed)
        output = main_model(data)
        # loss = loss_fn(output, target)
        prediction = torch.argmax(output, 1)
        correct_val += (prediction == target).sum().item()  # .float()
        batch_all += len(data)

valacc = correct_val/batch_all
print('（按acc）valacc:', valacc)