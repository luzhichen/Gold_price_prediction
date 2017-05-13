# -*- coding: UTF-8 -*-

import pandas as pd
from numpy import*
from preprocess_helper import*
from Prediction_helper import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


version = 4444
learn_rate = 0.001
epoch = 1200
split = 1000
batch = 100


if __name__ == "__main__":

    print('loading data...')
    data = pd.read_excel('./data/integration_v4.xlsx')
    data=data.set_index('datetime')

    print('preprocessing...')
    #数据预处理，函数代码见preprocess_helper.py
    dataset=dataset=generate_input_data(data,7,5)
    market_information=data[['Au_open','Au_close']]

    print('CNN...')
    # 训练CNN并返回测试集上的交易信号和预测结果，函数代码见Prediction_helper.py
    signal_CNN,result_CNN=train_net(dataset, version, learn_rate, epoch, batch,split)

    print('stimulating...')
    # 用CNN返回的交易信号进行回测，函数代码见Prediction_helper.py
    portfolio_CNN=market_stimulate(market_information,signal_CNN,10,'CNN_')

    print('SVM...')
    # 训练SVM并返回测试集上的交易信号和预测结果，函数代码见Prediction_helper.py
    signal_SVM,result_SVM=train_sklearn_classifier(SVC(C=100,kernel='sigmoid',gamma=0.005),dataset,split,'SVM')

    print('stimulating...')
    # 用SVM返回的交易信号进行回测，函数代码见Prediction_helper.py
    portfolio_SVM = market_stimulate(market_information, signal_SVM, 10, 'SVM_')

    print('LogisticRegression...')
    # 训练Logistic回归并返回测试集上的交易信号和预测结果，函数代码见Prediction_helper.py
    signal_LR,result_LR = train_sklearn_classifier(LogisticRegression(C=100,tol=0.01),dataset,split,'LR')

    print('LogiticRegression stimulating...')
    # 用LR返回的交易信号进行回测，函数代码见Prediction_helper.py
    portfolio_LR = market_stimulate(market_information, signal_LR, 10, 'LR_')

    #整合数据
    print('random stimulate...')
    signal_random = array([random.random() for i in range(len(signal_CNN))]) > 0
    signal_random = pd.DataFrame(signal_random * 1, index=signal_CNN.index, columns=['buy_signal'])
    portfolio_random = market_stimulate(market_information, signal_random, 10, 'random_')

    stimulation_result = pd.concat([portfolio_CNN, portfolio_SVM, portfolio_LR, portfolio_random], axis=1)
    stimulation_result.to_excel('./result/stimulation_result.xlsx')
    stimulation_result.plot(stimulation_result.index, stimulation_result.columns, figsize=(14, 10))
    plt.savefig('./result/result.png')

    result=pd.concat([result_CNN,result_SVM,result_LR])
    result.to_excel('./result/result_integration.xlsx')




