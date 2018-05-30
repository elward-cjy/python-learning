# -*- coding: utf-8 -*-
#特征值.py
import numpy as np
import os
import io
import sys
import pickle as pk

#配置UTF-8输出环境
#reload(sys)
#sys.setdefaultencoding('utf-8')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
#数据文件转矩阵
def file2matrix(path,delimiter):
    recordlist = []
    fp = open(path,"rb")
    content  = fp.read()
    fp.close()
    rowlist = content.splitlines()
    print (rowlist)
    recordlist = [map(eval,row.split(delimiter.encode(encoding = "utf-8")))for row in rowlist if(row.strip()) ]
    print("mat:\n",np.mat(recordlist))
    return np.mat(recordlist)

root = "testdata"
pathlist = os.listdir(root)
for path in pathlist:
    recordmat  = file2matrix(root+"/"+path,"\t")
    print(np.shape(recordmat))

file_obj = open(root+"/recordmat.dat","wb")
pk.dump(recordmat[0],file_obj)
file_obj.close()

read_obj = open(root+"/recordmat.dat","rb")
readmat = pk.load(read_obj)
print("readmat:\n",readmat)
print(np.shape(readmat))