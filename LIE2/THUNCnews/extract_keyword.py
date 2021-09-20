import jieba
import xlrd
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


trainxlsx = r'./data/datatrain.xlsx'
book = xlrd.open_workbook(trainxlsx, encoding_override='utf-8')
trainQues = book.sheet_by_index(0)

class_num = 14
count = 0
sendict = {str(item): [] for item in range(class_num)}
# sendict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": []}

for row in range(1, trainQues.nrows):
    row_data = trainQues.row_values(row)
    if row_data[0] not in sendict[str(int(row_data[1]))]:
        sendict[str(int(row_data[1]))].append(row_data[0].strip('\n'))
corpus = []  ###语料
for key in sendict:
    sendict[key] = ','.join(sendict[key])
    corpus.append(sendict[key])
# print(len(Corpus))

# #停词过滤
stopwords = open('./data/hit_stopwords.txt', encoding='utf-8')
stopwords_list = stopwords.readlines()
stopwords = [x.strip() for x in stopwords_list] #去掉每行头尾空白
# print(stopwords)
def tokenizer(s):
    words = []
    cut = jieba.cut(s)
    for word in cut:
        if word not in stopwords:
            words.append(word)
    words = ' '.join(words)
    return words

# 读取文件数据，分词
corpus_cut = []
for line in corpus:
    # row_data[0] = ''.join([i for i in row_data[0] if not i.isdigit()])
    line = ''.join([i for i in line if not i.isdigit()])
    # line = ''.join([i for i in line if not i.isalpha()])
    s = tokenizer(line.strip())
    corpus_cut.append(s)
# print(len(corpus_cut))
# print(corpus_cut)
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus_cut))
word=vectorizer.get_feature_names()
weight = tfidf.toarray()
# print(type(weight))
key_word = []
output_path='./data/weights0.txt'
with open(output_path,'w+',encoding='utf-8') as o:
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
            o.write("-------这里输出第"+ str(i) +"类文本的词语tf-idf权重------"+"\n")
            for j in range(len(word)):
                if(weight[i][j]>0.01):
                    # o.write(str(weight[i][j])+"\n")
                    o.write(word[j]+"\n")
# print(len(weight))



import openpyxl
wb = openpyxl.Workbook()
ws1 = wb.create_sheet()

for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    ws1.append([str(i)])
    for j in range(len(word)):
        if (weight[i][j] > 0.01):
            ws1.append([weight[i][j]])
            # ws1.append(weight[i][j],column_offset=0)
wb.save('./data/weights2.xlsx')