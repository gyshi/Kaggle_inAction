# -*- coding:utf-8 -*-
import pandas as pd
import  numpy as np
from pandas import Series, DataFrame

obj = pd.Series([4,7,-5,3])
obj2 = pd.Series([4,7,-5,3], index = ["a","b","c","d"])
data = pd.DataFrame(np.arange(12).reshape(3,4),index = ['a','v','w'],columns= ['o','21','a','2'])
a = np.arange(9)
print(a)