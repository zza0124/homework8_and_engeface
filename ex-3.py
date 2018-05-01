from calculated import CHAZHI
import numpy as np
import matplotlib.pylab as plt
from calculated import variance
import math
from calculated import CHAZHI
import os
import re
A=[1,1]
B=[4,5]
X=3
print(CHAZHI.chazhi(A,B,X))
y=[1,2,3,5,4,6,8,9,4]
print(y*2)
tim=2
im=np.array([[5,5,5,8,9,6],[4,5,6,8,7,9],[3,6,9,5,6,6],[7,8,9,6,5,4],[1,4,7,2,5,8],[9,6,3,2,5,8]])
sh=np.shape(im)
b=np.ones((sh[0],sh[1]),np.float64)
tim=int(tim)
b = np.ones((sh[0] *tim-tim+1, sh[1] *tim-tim+1), np.float64)
b[::tim,::tim]=im
def plti(im,**kwargs):
    plt.imshow(im,interpolation='none',**kwargs)
    plt.axis('off')
    plt.savefig('d:/new.jpeg')
    plt.show()
for i in range(sh[0]):
    for j in range(sh[1]-1):
        try:
            a1=[im[i,j-2],im[i,j-1],im[i,j],im[i,j+1]]
            s1=variance.variance(a1)
        except IndexError as e:s1=10000
        try:
            a2= [im[i , j-1], im[i , j], im[i, j+1], im[i , j+2]]
            s2 = variance.variance(a2)
        except IndexError as e:s2=10000
        try:
            a3 = [im[i, j], im[i , j+1], im[i , j+2], im[i , j+3]]
            s3 = variance.variance(a3)
        except IndexError as e:s3=10000
        if min(s1,s2,s3)==s1:
            imp=a1
            X=[j-2,j-1,j,j+1]
        if min(s1,s2,s3)==s2:
            imp = a2
            X = [j - 1, j ,  j + 1,j+2]
        if min(s1,s2,s3)==s3:
            imp = a3
            X = [j , j+1, j+2 , j + 3]
        if tim==2:
            print('X=',tim*X)
            print('j=',j)
            print('tim * i + 1=', tim * j + 1)
            b[tim*i,tim*j+1] = CHAZHI.lagrange(np.dot(X,tim ), imp, tim * j + 1)
        if tim>2:
            for k in range(tim-1):
                x0=np.add(np.dot(np.ones(tim-1),tim*i),list(range(tim-1)))
                b[tim*i:tim*i+tim,j]=CHAZHI.lagrange(tim*X,imp,x0)
b=b.T
for i in range((sh[0]*tim)-1):
    for j in range(sh[1]-1):
        try:
            a1=[b[i,j-2],b[i,j-1],b[i,j],b[i,j+1]]
            s1=variance.variance(a1)
        except IndexError as e:s1=10000
        try:
            a2= [b[i , j-1], b[i , j], b[i, j+1], b[i , j+2]]
            s2 = variance.variance(a2)
        except IndexError as e:s2=10000
        try:
            a3 = [b[i, j], b[i , j+1], b[i , j+2], b[i , j+3]]
            s3 = variance.variance(a3)
        except IndexError as e:s3=10000
        if min(s1,s2,s3)==s1:
            imp=a1
            X=[j-2,j-1,j,j+1]
        if min(s1,s2,s3)==s2:
            imp = a2
            X = [j - 1, j ,  j + 1,j+2]
        if min(s1,s2,s3)==s3:
            imp = a3
            X = [j , j+1, j+2 , j + 3]
        if tim==2:
            print('i=',i)
            b[i,tim*j+1] = CHAZHI.lagrange(np.dot(X,tim ), imp, tim * j + 1)
        if tim>2:
            for k in range(tim-1):
                x0=np.add(np.dot(np.ones(tim-1),tim*i),list(range(tim-1)))
                b[i:tim*i+tim,j]=CHAZHI.lagrange(tim*X,imp,x0)
print(b)
plti(b)