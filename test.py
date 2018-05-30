# -*- coding: utf-8 -*-
#特征值.py
import numpy as np
import pandas as pd
#import os
#import io
#import sys
import pickle as pk
import matplotlib.pyplot as plt
from  PIL import Image 
from math import sqrt 
import tensorflow as tf
import re
import matplotlib.pyplot as plt
from pylab import *

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#配置UTF-8输出环境
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
# x = linspace(0, 5, 10)
# y = x ** 2

# fig, ax = plt.subplots(figsize = (8,4),dpi = 100)

# ax.plot(x, x**2, label="y = x**2")
# ax.plot(x, x**3, label="y = x**3")
# ax.legend(loc=2); # upper left corner
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('title');

# plt.show()


#print(df.dtypes)
#print(df['Cabin'].isnull())

# x = np.arange(1,10)
# y = x
# fig = plt.figure()
# ax1 = fig.add_subplot(111)

# ax1.set_title('Scattter Plot')

# plt.xlabel('X')
# plt.ylabel('Y')
# ax1.scatter(x,y,c = 'b',marker = 'o')
# plt.legend('x1')
# plt.show()

# pil_im = Image.open('1.png')
# print(pil_im.size)
# pil_im.show()

# old = [1,2,3,4]
# new = old
# old = 6
# print("new:",new)
# print("old:",old)
# def func(A):
#     for itr in A:
#         yield itr

# for itr in func([1,2]):
#     print(itr)

# print(max("zhuanlan.zhihu.com/mBrain"))
# def func(A):
#     A[1]=6
# L = [1,2,3,4]
# func(L)
# print(L)

# def func(A):
#     A = 6
#     print(A)
# b=3
# func(b)
# print(b)
# print([lambda x:x*2 for x in range(3)])
# print([(lambda x:x*2) (x)for x in range(3)])

# for itr in enumerate([1,2]):
#     print(itr)

# for itr in zip([1,2],[6,7]):
#     print(itr)

# print([1,2]+[4,5])

# print(sqrt(3)*sqrt(3))

# match_test = re.match('/(.*)/(.*)/*','/usr/home/jjj')
# print(match_test.groups())


# A = [[8,1,6],[3,5,7],[4,9,2]]
# evals,evecs = np.linalg.eig(A)
# print("特征值:\n",evals,"\n特征向量:\n",evecs)
# sigma = evals*np.eye(3)
# print(sigma)
# print(np.linalg.inv(evecs.T)*sigma*evecs.T)

# vectormat = np.mat([[1,2,3],[4,5,6]])
# varmat = np.std(vectormat.T,axis = 0)
# normvmat = (vectormat-np.mean(vectormat)/varmat.T)
# normv12 = normvmat[0] - normvmat[1]

# print("varmat:\n",varmat)
# print(sqrt(normv12*normv12.T))


# v12 = vectormat[0] - vectormat[1]
# print(sqrt(v12*v12.T))

# matrix = np.mat([1,2,3,4])
# print(matrix)
# print(matrix.T)
# S1 =  {1,2,3,4}
# print(S1<{1,2,3,4})

# line = "theknightswho"
# print(line.isalpha())

# x = 1.23456789
# print("%s " % x,str(x))
# site = {"name": "菜鸟教程", "url": "www.runoob.com"}
# print("网站名：{name}, 地址 {url}".format(**site))

# # template = '{motto},{pork},{food}'
# # print(template.format({'motto':'spam','pork':'ham','food':'eggs'}))

# somelist = {'s':1,'q':2,'u':3,'y':4}
# print('first={s},third={u}'.format(**somelist))
