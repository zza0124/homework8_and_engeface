import sympy
# x=sympy.symbols('x')
# print(sympy.solve(x**3-x-1))
# pi=3.1415926535
def func(a):
    return(-a**3+1.5)
def dichotomy(func,a,b,xi):
    if a>b:
        tem=b
        b=a
        a=tem
    if a==b:
        print('输入参数错误')
    x1=func(a)
    x2=func(b)
    k=0
    while x1*x2>=0:
        print(x1,x2)
        print('结果不在范围，请重新输入')
        a=input('a=')
        b = input('b=')
        x1 = func(a)
        x2 = func(b)
    if x1*x2==0:
        if x1==0:
            print(a,k)
        else:
            print(b,k)
    else:
        while abs(a-b)>xi:
            m = (a + b) / 2
            if func(a)*func(m)<0:
                b=m
            else:a=m
            k=k+1
        print(m, k)
    return