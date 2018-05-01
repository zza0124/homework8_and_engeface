def chazhi(a,b,c):
    return a[1]-(a[1]-b[1])/(a[0]-b[0])*a[0]+c*(a[1]-b[1])/(a[0]-b[0])*a[0]
def lagrange(x,y,x0):
    n=len(x)
    if (type(x0)!=int):
        m=len(x0)
        s=list(range(m))
        for k in range(m):
            t=0
            for j in range (n):
                u=1.0
                for i in range (n):
                    if i !=j:
                        u=u*(x0[k]-x[i])/(x[j]-x[i])
                t=t+u*y[j]
            s[k]=t
        return s
    else :
        t=0
        for j in range (n):
            u=1.0
            for i in range (n):
                if i !=j:
                    u=u*(x0-x[i])/(x[j]-x[i])
            t=t+u*y[j]
    return t