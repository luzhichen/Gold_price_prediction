# -*- coding: UTF-8 -*-

from keras.models import Sequential,Model
from keras.layers import *
import pandas as pd
from keras.optimizers import SGD
from numpy import*
import json
import pandas as pd
import random
from keras.callbacks import*
from keras.optimizers import*

dropout_prob = (0.25, 0.5)
conv_poo_len=[[2,2]]
chanel=8
momentum_=0.9
hidden_dims=8

#从预处理好的数据集中获取训练集和测试集
def load_data(dataset, split):
    feature = dataset['feature']
    feature = feature.reshape(feature.shape[0], 1, feature.shape[1], feature.shape[2])
    label = dataset['label_2_class']
    x_train = feature[:split]
    y_train = label[:split]
    x_test = feature[split:]
    y_test = label[split:]

    return [x_train, y_train, x_test, y_test]

#搭建CNN网络，输入为学习率、输入数据的长（时间窗口）、宽（特征数量）
def build_network(lrate,col,row):
    model = Sequential()
    graph_in = Input(shape=(1, row, col))

    convs = []
    for conv_l in conv_poo_len:
        conv = Convolution2D(chanel, conv_l[0], col, border_mode='valid', dim_ordering='th')(graph_in)
        act = Activation('sigmoid')(conv)
        pool = MaxPooling2D(pool_size=(1, conv_l[1]))(act)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(convs)>1:
        out = Merge(mode='concat')(convs)
    else:
        out=convs[0]
    graph = Model(input=graph_in, output=out)
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=lrate, decay=1e-6, momentum=momentum_, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


#计算精准率、召回率、F1值和准确率
def calcu_indicate(y_test,y_pre,which_class):
    corre=(y_pre == which_class)* (y_test == which_class)
    recall = float(sum(corre)) / sum(y_test == which_class)
    precise = float(sum(corre)) / sum(y_pre == which_class)
    f1 = 2 / ((1 / recall) + (1 / precise))

    return [precise,recall,f1]
#整合数据
def integrate_result(y_test,y_pre,acc,clf_name):
    pos_ind = calcu_indicate(y_test, y_pre, which_class=1)
    print('precise_pos:' + str(pos_ind[0]) + ' recall_pos:' + str(pos_ind[1]) + ' F1_pos:' + str(pos_ind[2]))
    neg_ind = calcu_indicate(y_test, y_pre, which_class=0)
    print('precise_neg:' + str(neg_ind[0]) + ' recall_neg:' + str(neg_ind[1]) + ' F1_neg:' + str(neg_ind[2]))
    result = pd.DataFrame(array([pos_ind, neg_ind]), index=[clf_name+'_pos', clf_name+'_neg'], columns=['precise', 'recall', 'F1'])
    result['accuracy']=acc

    return result

#根据给定参数（学习率、batch大小、训练次数、测试集、训练集分割比例）搭建并训练网络
def train_net(dataset,version,learn_rate,epoch,batch,split):

    name='v'+str(version)+'_lr'+str(learn_rate)+'_epoch'+str(epoch)
    x_train, y_train, x_test, y_test=load_data(dataset,split)

    row = x_train.shape[2]
    col = x_train.shape[3]


    assert(len(x_train)==len(y_train))

    model=build_network(learn_rate,col=col,row=row)
    csv_logger = CSVLogger("./log/" + name+'.log')
    model.fit(x_train, y_train, nb_epoch=epoch, batch_size=batch, callbacks=[csv_logger],validation_split=0.2 )#validation_data=[x_test,y_test]

    model.save_weights('./net_conf/'+name+'.h5')
    conf_json = model.get_config()

    with open('./net_conf/'+name+'.json', 'w') as json_file:
        json_file.write(json.dumps(conf_json))

    y_pre = model.predict_classes(x_test)
    y_pre = y_pre.reshape(y_pre.shape[0], )
    eva=model.evaluate(x_test,y_test)
    print('CNN loss&accuracy:')
    print eva

    result=integrate_result(y_test,y_pre,eva[1],clf_name='CNN')
    #result.to_excel('./result/CNN_'+name+'.xlsx')

    trade_signal = pd.DataFrame(y_pre, index=dataset['timestamp'][split:], columns=['buy_signal'])

    # 输出交易信号和预测结果
    return trade_signal,result




#根据给定的数据集和分类器训练sklearn模块中的分类器（SVM、Logistic回归）
def train_sklearn_classifier(clf,dataset,split,clf_name):


    x_train, y_train, x_test, y_test = load_data(dataset, split)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[-2]*x_train.shape[-1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[-2] * x_test.shape[-1])
    clf.fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    score = clf.score(x_test,y_test)
    y_pre = y_pre.reshape(y_pre.shape[0], )
    print('accuracy:'+str(score))

    result = integrate_result(y_test, y_pre,score, clf_name)
    #result.to_excel('./result/'+clf_name+'_result'+ '.xlsx')
    trade_signal = pd.DataFrame(y_pre, index=dataset['timestamp'][split:],columns=['buy_signal'])
#输出交易信号和预测结果
    return trade_signal,result

#根据给定交易信号进行市场模拟策略回测
def market_stimulate(market_information,trade_signal,order_share,mark):
    market_information = market_information[trade_signal.index[0]:]
    stimulate = pd.concat([market_information, trade_signal], axis=1)
    stimulate['sell_signal'] = -1 * stimulate['buy_signal'].shift(5)
    stimulate = stimulate.fillna(0)
    stimulate['cash'] = 0
    stimulate['Au_value'] = 0
    stimulate['buy_signal'] = stimulate['buy_signal'] * order_share
    stimulate['sell_signal'] = stimulate['sell_signal'] * order_share

    for i in range(1, len(stimulate)):
        open_ = stimulate.loc[stimulate.index[i], 'Au_open']
        close_ = stimulate.loc[stimulate.index[i], 'Au_close']
        buy = stimulate.loc[stimulate.index[i], 'buy_signal']
        sell = stimulate.loc[stimulate.index[i], 'sell_signal']
        stimulate.loc[stimulate.index[i], 'Au_value'] = (buy + sell) * close_ + stimulate.loc[stimulate.index[i - 1], 'Au_value']
        stimulate.loc[stimulate.index[i], 'cash'] = stimulate.loc[stimulate.index[i - 1], 'cash'] - sell * close_ - buy * open_

    stimulate[mark+'portfolio'] = stimulate['cash'] + stimulate['Au_value']

    #输出资产组合历史变化
    return stimulate[[mark+'portfolio']]







if __name__ == "__main__":
    dataset=pd.read_pickle("./data/dataset_v4.pk")
    version=4444
    learn_rate=0.001
    epoch=1200
    split=1000
    batch = 100
    train_net(dataset, version, learn_rate, epoch, batch,split)














