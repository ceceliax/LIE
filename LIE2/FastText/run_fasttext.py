# coding: UTF-8
import time
import torch
import numpy as np
from FastText.train_eval import train
from importlib import import_module
from FastText.utils_fasttext import build_dataset, build_iterator
from FastText.fasttext import Model
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
lr = 0.0002
pad_size = 32
n_gram_vocab = 250499
num_class = 31

if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


    print("Loading data...")
    vocab_path = './data/vocab.pkl'
    train_path = 'F:\研二\LIE\BERT\data\chinese_intent_train.txt'
    dev_path = 'F:\研二\LIE\BERT\data\chinese_intent_dev.txt'
    test_path = 'F:\研二\LIE\BERT\data\chinese_intent_test.txt'
    save_path2 = './data/fasttext2.ckpt'
    vocab, train_data, dev_data, test_data = build_dataset(vocab_path, train_path, dev_path, test_path, pad_size, n_gram_vocab, use_word=False)
    train_iter = build_iterator(train_data, batch_size,device)
    dev_iter = build_iterator(dev_data, batch_size,device)
    test_iter = build_iterator(test_data, batch_size,device)

    model = Model(num_classes=num_class).cuda()
    train(model, train_iter, dev_iter, test_iter, learning_rate=lr)

    from FastText.train_eval import evaluate
    model.load_state_dict(torch.load(save_path2))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
