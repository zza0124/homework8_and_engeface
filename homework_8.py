# 初始化
import pandas as pd
import numpy as np
import time
import matplotlib.pylab as plt
from pylab import mpl
from datetime import datetime
from calculated import average
from calculated import cha
from calculated import CHAZHI
from sklearn import datasets, linear_model
#画图
mpl.rcParams['font.sans-serif'] = ['SimHei']
df = pd.read_csv("data1.csv")
s=np.shape(df)
tim=1
x=list(range(s[0]*tim-tim+1))
plp=cha.cha(df.__array__()[:,1],tim)
plp=plp[::-1]
pap=cha.cha(df.__array__()[:,2],tim)
pap=pap[::-1]
ulp=cha.cha(df.__array__()[:,3],tim)
ulp=ulp[::-1]
uap=cha.cha(df.__array__()[:,4],tim)
uap=uap[::-1]
x=list(range(len(plp)))
fig = plt.figure(figsize=(16, 8))
k=df.__array__()[:,0]
coloall = ['red', 'blue', 'green', 'orange', 'yellow', 'purple', 'black']
line1 = plt.plot(x, plp, label="个人最低成交价格", color=coloall[1], linewidth=0.5)
line2 = plt.plot(x, pap, label="个人平均成交价格", color=coloall[2], linewidth=0.5)
line3 = plt.plot(x, ulp, label="单位最低成交价格", color=coloall[3], linewidth=0.5)
line4 = plt.plot(x, uap, label="单位平均成交价格", color=coloall[4], linewidth=0.5)
plt.xlabel("时间(第0天为2012年8月28日)/月")
plt.ylabel("价格/元")
plt.title("交易价格走势图(关掉此图程序继续运行)")
plt.show()
#预测
k=4
xp=x[-k:]
yplp=plp[-k:]
x0=xp[-1]+1
print(yplp)
predict1=CHAZHI.lagrange(xp,yplp,x0)
print('预测的个人最低成交价格是',predict1)
ypap=pap[-k:]
print(ypap)
predict2=CHAZHI.lagrange(xp,ypap,x0)
print('预测的个人平均成交价格是',predict2)
####
# 计算政策影响参数
k=4
xp=x[-k:-1]
yplp=plp[-k:-1]
x0=xp[-1]+1
print(yplp)
predict1=CHAZHI.lagrange(xp,yplp,x0)
theta1=plp[-1]-predict1
print('预测的个人最低成交价格的政策影响参数为',theta1)
ypap=pap[-k:-1]
print(ypap)
predict2=CHAZHI.lagrange(xp,ypap,x0)
theta2=pap[-1]-predict2
print('预测的个人平均成交价格的影响参数是',theta2)
plp[-1]=predict1
pap[-1]=predict2
###修正后的重新预测
k=4
xp=x[-k:]
yplp=plp[-k:]
x0=xp[-1]+1
print(yplp)
predict1=CHAZHI.lagrange(xp,yplp,x0)
print('修正后预测的个人最低成交价格是',predict1+theta1)
ypap=pap[-k:]
print(ypap)
predict2=CHAZHI.lagrange(xp,ypap,x0)
print('修正后预测的个人平均成交价格是',predict2+theta2)
