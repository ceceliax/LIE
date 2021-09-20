import xlrd
import jieba
from sklearn.model_selection import train_test_split




trainxlsx = r'./data/datatrain.xlsx'
# testxlsx = r'./data/datatest.xlsx'

#### 生成停用词列表
stopwords = open('./data/hit_stopwords.txt', encoding='utf-8')
stopwords_list = stopwords.readlines()
stopwords = [x.strip() for x in stopwords_list]    #去掉每行头尾空白

### 用于分离数据和标签,并去除数据中的停用词
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


data, label = all_data(trainxlsx)

x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True, stratify=label)

# for i in range(len(x_train)):
#     temp = x_train[i]+"\t"+str(y_train[i])
#     f = open('./data/traindata.txt', 'a', encoding='utf-8')
#     f.write(temp + '\n')
#     f.close()

