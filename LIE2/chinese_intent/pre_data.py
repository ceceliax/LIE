import xlrd
import jieba



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

trainxlsx = r'F:\研二\LIE\chinese_intent\data\train.xlsx'
row_data, row_label = all_data(trainxlsx)

for i in range(len(row_data)):
    w = open('./data/train.txt', 'a', encoding='utf-8')
    w.write(row_data[i]+'\t' + str(row_label[i]) + '\n')
    w.close()