# -*- coding: utf-8 -*-
import pandas as pd #数据分析
import matplotlib.pyplot as plt
import csv
import data_pro
import sys
import io
from numpy import  *
#配置UTF-8输出环境
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#测试的时候路径可能要改
csvfile0 = pd.read_csv(".\\homework\\week1\\Titanic_bayes\\train.csv")
#每次迭代训练集测试数目可调
trainPredictlen = 20
#迭代验证次数，可调
iterint = 10

train_data = []
tmp_train_data = []
trainSize = csvfile0.shape[0]

tmp_train_predict = []
train_predict = []
sum_of_error = 0
for iteration  in range(iterint):
    random_list = []
    #生成随机数
    for i in range(trainPredictlen):
            random_list.append(random.randint(0, trainSize))

    #print("randlist:\n",random_list)
    tmp_train_label = []
    for index in random_list:
        tmp_train_predict = data_pro.predict_passenger(csvfile0['Pclass'][index],csvfile0['Sex'][index],csvfile0['Age'][index],csvfile0['SibSp'][index],csvfile0['Parch'][index],csvfile0['Fare'][index],csvfile0['Embarked'][index])
        train_data.append(tmp_train_predict)
        tmp_train_label.append(csvfile0['Survived'][index])
    switchlabel = {0:'nosurvived',1:'survived'}
    train_label = []
    for i in range(trainPredictlen):
        train_label.append(switchlabel[tmp_train_label[i]])
    #print("train_label:\n",train_label)
        
    error_count = 0
    #print("error_count:\n",error_count)
    for turn in range(trainPredictlen):
        #print("train_data:\n",train_data[turn])
        if train_data[turn] != train_label[turn]:
            error_count += 1
            #print("error_count:\n",error_count)
    sum_of_error += error_count
    #print("sumerror:\n",sum_of_error)         
    print("第%d次训练集验证准确率为：%f\n" % (iteration,(1-error_count/trainPredictlen)))
final_rate = 1 - sum_of_error/(iterint*trainPredictlen)
#print("final_rate:\n",final_rate)
print("训练集%d次测试的平均准确率为：%f" % (iterint,final_rate))

#测试集测试数目可调
test_size = 10
test_data = []
tmp_test_data = []
random_list = []
csvfile = pd.read_csv('.\\homework\\week1\\Titanic_bayes\\test.csv',nrows = test_size)
test_allnum = csvfile.shape[0]


#生成随机数
for i in range(test_size):
        random_list.append(random.randint(0, test_allnum))
for i in random_list:
    tmp_test_data = [csvfile['Pclass'][i],csvfile['Sex'][i],csvfile['Age'][i],csvfile['SibSp'][i],csvfile['Parch'][i],csvfile['Fare'][i],csvfile['Embarked'][i]]
    test_data.append(tmp_test_data)
test_predict = []
tmp_test_predict = []
for turn in range(test_size):
    tmp_test_predict = data_pro.predict_passenger(test_data[turn][0],test_data[turn][1],test_data[turn][2],test_data[turn][3],test_data[turn][4],test_data[turn][5],test_data[turn][6])    
    test_predict.append(tmp_test_predict)

print("测试集预测结果为:\n",test_predict)

'''
#下面是训练集各特征对获救情况的影响分布图
#有中文出现的情况，需要u'内容'
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
csvfile0.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not.
plt.title(u"获救情况 (1为获救)") # puts a title on our graph
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))
csvfile0.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(csvfile0.Survived, csvfile0.Age)
plt.ylabel(u"年龄")                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((2,3),(1,0), colspan=2)
csvfile0.Age[csvfile0.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
csvfile0.Age[csvfile0.Pclass == 2].plot(kind='kde')
csvfile0.Age[csvfile0.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
csvfile0.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

'''





