import numpy as np
import torch
from BERT.utils import build_dataset, build_iterator, get_news_label_embed
from BERT.bert import Model
from transformers.optimization import AdamW
import torch.nn.functional as F
from sklearn import metrics
import time
import os
import random
from datetime import timedelta
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 1
seed_torch(seed)
num_class = 31
pad_size = 32                                              # 每句话处理成的长度(短填长切)
max_sen_len = 32

batch_size = 64
learning_rate = 1e-6

hidden_size = 768 ###embed_dim
Embed_dim = 768
num_epochs = 300
require_improvement = 1000                            # 若超过1000batch效果还没提升，则提前结束训练

save_path = './data/bert.ckpt'
save_path2 = './data/bert2.ckpt'
save_path3 = './data/bert3.ckpt'
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True  # 保证每次结果一样
class_list = [x.strip() for x in open('./data/chinese_intent_class.txt',encoding='utf-8').readlines()]
# class_list = [x.strip() for x in open('./data/snips_class.txt').readlines()]

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


print("Loading data...")
train_data, dev_data, test_data = build_dataset(pad_size)
train_iter = build_iterator(train_data, batch_size)
dev_iter = build_iterator(dev_data, batch_size)
test_iter = build_iterator(test_data, batch_size)
label_embed = get_news_label_embed(max_sen_len).cuda()



# bert_path = '.\\data\\uncased_L-12_H-768_A-12'
bert_path = '.\data\BERT_cased_L-12_H-768_A-12'


model = Model(bert_path=bert_path,
              hidden_size=hidden_size,
              num_classes=num_class,
              label_embed=label_embed,
              embed_dim=Embed_dim,
              atten_size=30,
              label_hidden_size=384,
              max_sen=max_sen_len
              ).cuda()

def train( model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    lr_list = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.9)
    model.train()
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(model.parameters(),
                         lr=learning_rate)#, eps=1e-8)#, weight_decay=1e-3,betas=(0.9,0.99))
    total_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(num_epochs):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                test_, test_loss = evaluate(model, test_iter)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), save_path2)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},Test Acc: {5:>6.2%}, Time: {6} {7}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, test_,improve, time_dif))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
                #         # 验证集loss超过1000batch没下降，结束训练

        if flag:
            break
    torch.save(model.state_dict(), save_path3)
    test(model, test_iter)

def test( model, test_iter):
    # test
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

train( model, train_iter, dev_iter, test_iter)


model.load_state_dict(torch.load(save_path2))
model.eval()
test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
print(msg.format(test_loss, test_acc))
print("Precision, Recall and F1-Score...")
print(test_report)
print("Confusion Matrix...")
print(test_confusion)

# print("last epoch:")
# # model.load_state_dict(torch.load(save_path3))
# # model.eval()
# # test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
# # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
# # print(msg.format(test_loss, test_acc))
# # print("Precision, Recall and F1-Score...")
# # print(test_report)
# # print("Confusion Matrix...")
# # print(test_confusion)