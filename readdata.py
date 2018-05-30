# -*- coding: utf-8 -*-
#特征值.py
import numpy as np
import os
import io
import sys
import pickle as pk
#配置UTF-8输出环境
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
x = np.inspace(-5,5,200)