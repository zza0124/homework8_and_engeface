from calculated import dichotomc
import sympy
import numpy as np
import matplotlib as pib
def func(a):
     return(a**3-5)
k=dichotomc.dichotomy(func,-5,5,0.001)
print(k)