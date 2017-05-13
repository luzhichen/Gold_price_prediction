# -*- coding: UTF-8 -*-

import pandas as pd
from numpy import*
import random
import datetime
import talib

#样本标准化
def normalize(data_):
    data=data_.fillna(0)
    data_standard=pd.DataFrame(index=data.index)

    #每一个自序列减均值除与标准差
    for i in range(len(data.columns)):
        m=mean(data[data.columns[i]])
        v=std(data[data.columns[i]])

        if v==0:
            new_column = data[data.columns[i]].apply(lambda x: 0)
        else:
            new_column=data[data.columns[i]].apply(lambda x:(x-m)/v)
        data_standard=pd.concat([data_standard,pd.DataFrame({data.columns[i]:new_column})],axis=1)
    return data_standard
#fill nan value with latest avaliable value

#空值处理，延续前一时刻值
def fill_nan(data):
    for each in data.columns:
        if_nan=list(isnan(data[each]))
        while sum(if_nan) > 0:
            ind=if_nan.index(True)
            data.loc[data.index[ind],each]=data.loc[data.index[ind-1],each]
            if_nan = list(isnan(data[each]))
        return data



#根据未来收益方向对数据标注，正类标注为1负类为0
def label_data(data, forward):
    MA = pd.Series(talib.MA(data.Au_close.values, timeperiod=forward), index=data.index)

    profit = (MA.shift(-forward)-data['Au_close'])/data['Au_close']
    profit=profit.fillna(0)
    one_third=profit[int(0.333333*len(profit))]
    two_third=profit[int(0.666666*len(profit))]
    label_2_class=(profit>0)*1
    label_3_class=(profit>one_third)*1+(profit>two_third)*1
    return [label_2_class,label_3_class,profit]

#将原始数据预处理成输入分类器训练的格式
def generate_input_data(raw_data, slide_window, forward):
    data=fill_nan(raw_data)

    label_2_class, label_3_class, profit=label_data(data,forward)
    y_2_class=label_2_class.values[slide_window-1:-forward]
    y_3_class = label_3_class.values[slide_window - 1:-forward]
    x = []
    for i in range(slide_window-1,len(data)-forward):
        piece=normalize(data[i - slide_window + 1:i + 1])
        x.append(piece.values.tolist())
    assert (len(x) == len(y_2_class))
    assert (len(x) == len(y_3_class))
    dataset=dict()
    dataset['feature']=array(x)
    dataset['label_2_class']=y_2_class
    dataset['label_3_class'] = y_3_class
    dataset['timestamp']=profit.index[slide_window-1:-forward]
    dataset['profit']=profit[slide_window-1:-forward]

    return dataset


if __name__ == "__main__":
    data=pd.read_excel('./data/integration_v4.xlsx')
    data = data.set_index('datetime')
    #data=data[['Au_open','Au_close','Au_high','Au_low','Au_amount','Au_volume']]
    dataset=generate_input_data(data,7,5)
    pd.to_pickle(dataset,'./data/dataset_v4.pk')


