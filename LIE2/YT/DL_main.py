from YT.process_data import all_data,get_one2many_YT_label_embed,data_embed,get_all_sen_align,get_dataloder
from sklearn.model_selection import train_test_split
from model.LSTM import BP_LSTM
from model.atten_lstm import atten_lstm
# from model.attn_textCNN import CNN_Text
# from YT.transformer import transformer
from model.multi_atten import multi_atten_lstm

# from model.textCNN import CNN_Text
# from model.textcnn_new import CNN_Text
import torch

##参数
Embed_dim = 256
max_sen_len = 28
batch_size = 128
learning_rate = 0.0005    ## 学习率
seed = 1
cuda_able = True
dropout = 0.5
hidden_size = 128    ##模型隐藏层维度
bidirectional = True
output_size = 9   ##模型输出层维度
num_class = 9


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

data, label = all_data()    ##  data是已经分词好的问题
label_embed = get_one2many_YT_label_embed(max_sen_len).cuda()

####  先划分训练集测试集验证集 6：2：2，再嵌入,最后对齐，处理成相同的句子长度
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=44, stratify=label)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=44, stratify=y_train)




x_train_embed = data_embed(x_train)
x_test_embed = data_embed(x_test)
x_val_embed = data_embed(x_val)

train_embed_align = get_all_sen_align(x_train_embed, max_sen_len)  # print(all_sen_embed_align)  类型是list  1402个
val_embed_align = get_all_sen_align(x_val_embed, max_sen_len)
test_embed_align = get_all_sen_align(x_test_embed, max_sen_len)

train_dataloader = get_dataloder(train_embed_align, y_train, batch_size)
val_dataloader = get_dataloder(val_embed_align, y_val, batch_size)
test_dataloader = get_dataloder(test_embed_align,y_test, batch_size)
path = 'checkpoint_textCNN_1.pt'
log_dir = 'textCNN.path'
#####################################################################
###  构建网络  ###
#
# model = BP_LSTM(output_size=output_size,
#                 hidden_size=hidden_size,
#                 embed_dim=Embed_dim,
#                 bidirectional=bidirectional,
#                 dropout=dropout,
#                 sequence_length=max_sen_len).cuda()

# model = atten_lstm(output_size=output_size,
#                    hidden_size=hidden_size,
#                    embed_dim=Embed_dim,
#                    attention_size=32,
#                    sequence_length=max_sen_len,
#                    dropout=dropout)
# #
# model = multi_atten_lstm(output_size=output_size,
#                          hidden_size=hidden_size,
#                          embed_dim=Embed_dim,
#                          sequence_length=max_sen_len,
#                          atten_size=32,
#                          label_embed=label_embed,
#                          num_class=num_class,
#                          label_max_sen=max_sen_len,
#                          dropout=dropout).cuda()

## 设置保存模型的路径
from model.textCNN import CNN_Text
# from model.textcnn_new import CNN_Text

model = CNN_Text(kernel_num=100,
                 kernel_sizes=[3, 4, 5],
                 class_num=num_class,
                 embed_dim=Embed_dim,
                 # max_sen_len=max_sen_len,
                 # label=label_embed,
                 # atten_size=32,
                 dropout=dropout).cuda()



# model = transformer(output_size=output_size,
#                     embed_dim=Embed_dim,
#                     nhead=2,
#                     num_layer=2,
#                     sequence_length=max_sen_len)


### 实例化网络
loss_fn = torch.nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#### 早停策略
from model.pytorchtools import EarlyStopping
# Train the Model using Early Stopping
patience = 100	# 当验证集损失(或准确率？)在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience,path=path, verbose=True)	# 关于 EarlyStopping 的代码可先看博客后面的内容
n_epochs = 500

best_valacc = 0
best_loss = 100

### 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
for epoch in range(1,n_epochs+1):
    ##建立训练数据的dataloader
    model.train()
    train_loss = 0
    ##按小批量进行训练
    for batch,(data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    correct_val = 0
    valid_loss = 0
    batch_all = 0
    for batch, (data, target) in enumerate(val_dataloader):
        optimizer.zero_grad()
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        loss = loss_fn(output, target)
        valid_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct_val += (prediction == target).sum().item()  # .float()
        batch_all +=len(data)

    valid_loss = valid_loss/batch_all
    valacc = correct_val/batch_all
    print('valacc:',valacc)
    if valacc > best_valacc:
    # if valid_loss < best_loss:
        # lstm.load_state_dict(torch.load('checkpoint.pt'))  ###保存模型
        best_valacc = valacc
        # best_loss = valid_loss
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log_dir)
    early_stopping(valid_loss, model)
    # early_stopping(valacc, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print(epoch)


### 测试1

checkpoint = torch.load(path)
model.load_state_dict(checkpoint)

correct_test = 0
total_test = 0
model.eval()
# for test, test_label in test_dataloader:
for batch, (data, target) in enumerate(test_dataloader):
    data = data.cuda()
    target = target.cuda()
    # out = model(test, label_embed)
    out = model(data)
    # 计算准确率
    prediction = torch.argmax(out, 1)
    correct_test += (prediction == target).sum().item()  # .float()
    total_test += len(target)
print('（按loss）testacc: {:.6f}'.format(correct_test / total_test))
# print(epoch)

###测试2
checkpoint = torch.load(log_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']

model.eval()
correct_val = 0
batch_all = 0
for batch, (data, target) in enumerate(test_dataloader):
    data = data.cuda()
    target = target.cuda()
    optimizer.zero_grad()
    # output = model(data, label_embed)
    output = model(data)
    loss = loss_fn(output, target)
    prediction = torch.argmax(output, 1)
    correct_val += (prediction == target).sum().item()  # .float()
    batch_all += len(data)

valacc = correct_val/batch_all
print('（按acc）valacc:', valacc)

# ### 测试
#
# # dir = 'checkpoint.pt'
# # checkpoint = torch.load(dir)
# # model.load_state_dict(checkpoint)
#
#
# checkpoint = torch.load(log_dir)
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# epochs = checkpoint['epoch']
#
#
# correct_test = 0
# total_test = 0
# model.eval()
# for test, test_label in test_dataloader:
#     out = model(test, label_embed)
#     # out = model(test)
#     # 计算准确率
#     # out = out.reshape(len(im),9)
#     prediction = torch.argmax(out, 1)
#     correct_test += (prediction == test_label).sum().item()  # .float()
#     total_test += len(test_label)
# print('testacc: {:.4f}'.format(correct_test / total_test))
# print(total_test)
# print(correct_test)
# # # print(epoch)
#
# model.eval()
# correct_val = 0
# batch_all = 0
# for batch, (data, target) in enumerate(test_dataloader):
#     optimizer.zero_grad()
#     output = model(data,label_embed)
#     # output = model(data)
#     loss = loss_fn(output, target)
#     prediction = torch.argmax(output, 1)
#     acc = (prediction == target).sum().item()  # .float()
#     correct_val +=acc
#     # print('batch:',batch,'  len(data):', len(data))
#     batch_all += len(data)
#

# valacc = correct_val/batch_all
# print('valacc:', valacc)
# print(batch_all)
# print(correct_val)


