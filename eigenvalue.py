#特征值.py
import numpy as np
import io
import sys
# 改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
A = [[8,1,6],[3,5,7],[4,9,2]]
evals,evecs = np.linalg.eig(A)
print("特征值:",evals)
print("特征向量:",evecs)

sigma = evals*np.eye(3)
print(sigma)
B = evecs*sigma
print ("B:\n",B)
C = np.linalg.inv(evecs)
print("C:\n",C)
D = B*C
print("D:\n",D)