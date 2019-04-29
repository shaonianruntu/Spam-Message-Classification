import os 
import sys 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#读取输入数据，并返回相应的分词后的特征集和标识集
def get_data(data_file_path):
    
    lines = []    
    labels = []
    
    for line in open(data_file_path, "r", encoding="utf-8"):
        #对每一个行样本信息按照制表符进行分割得到list
        #第一行为标签，第二行为原短信内容，第三行为分词处理之后的短信内容
        arr = line.rstrip().split("\t")
        #如果样本信息不完整（不足三行）就丢弃
        if len(arr) < 3:
            continue
        
        #读取标签，并当负标签为-1时，对其进行转化为0
        if int(arr[0]) == 1:
            label = 1
        elif int(arr[0]) == 0 or int(arr[0]) == -1:
            label = 0
        else:
            continue
        labels.append(label)
        
        #读取分词之后的句子
        line = arr[2].split()
        lines.append(line)
        
    return lines, labels


#创建词袋字典
def create_vocab_dict(data_lines):
    vocab_dict = {}
    for data_line in data_lines:
        for word in data_line:
            if word in vocab_dict:
                vocab_dict[word] += 1
            else:
                vocab_dict[word] = 1
    return vocab_dict


#得到输入分词后的样本的BOW特征
def BOW_feature(vocab_list, input_line):
    return_vec = np.zeros(len(vocab_list), )
    for word in input_line:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


#主函数
if __name__ == "__main__":

    #数据集文件地址
    train_file_path = "mudou_spam.train"
    test_file_path = "mudou_spam.test"

    #得到分离后的特征和标记
    train_data, train_label = get_data(train_file_path)
    test_data, test_label = get_data(test_file_path)

    #构建文本的BOW词袋特征
    #BOW字典
    vocab_dict = create_vocab_dict(train_data)
    #对字典按照value进行排序
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda d:d[1], reverse=True)
    #筛出字典中value小于min_freq的键值对，并生成相应的键队列，得到BOW特征
    min_freq = 5
    vocab_list = [v[0] for v in sorted_vocab_list if int(v[1]) > min_freq]
   
    #生成文本的BOW特征
    train_X = []
    for one_msg in train_data:
        train_X.append(BOW_feature(vocab_list, train_data))

    test_X = []
    for one_msg in test_data:
        test_X.append(BOW_feature(vocab_list, test_data))

    #将数据格式转化为 numpy.array 格式
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    #训练模型
    model = LogisticRegression()
    model.fit(train_X, train_label)
    pred = model.predict(test_X)

    #模型评估
    train_acc = model.score(train_X, train_label)
    print("Train accuracy: ", train_acc)
    test_acc = model.score(test_X, test_label)
    print("Test accuracy: ", test_acc)

    #模型预测
    pred_prob_y = model.predict_proba(test_X)[:, 1]
    test_auc = metrics.roc_auc_score(test_label, pred_prob_y)
    print("Test AUC: ", test_auc)