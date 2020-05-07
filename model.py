import pandas as pd
import numpy as np
import matplotlib.pyplot as ptp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv('F:\SONGS\MachineLearning-master\MachineLearning-master\Datasets\Salary_Data.csv')
data.head()
real_x=data.iloc[:,0:1]
real_y=data.iloc[:,1]
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.3,random_state=0)
ln=LinearRegression()
ln.fit(train_x,train_y)
pr=ln.predict(test_x)
pr
test_y
import pickle
pickle.dump(ln,open('model.pkl','wb'))
