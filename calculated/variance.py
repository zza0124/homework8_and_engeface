#出于做出一个普适性强的函数计算包的目的，此处并未采用论文中的简化计算的方法
def variance(a):
    L=len(a)
    aver=sum(a)/L
    a1=list(range(L))
    for i in range(L):
        a1[i]=(a[i]-aver)**2
    var=sum(a1)/L
    return var