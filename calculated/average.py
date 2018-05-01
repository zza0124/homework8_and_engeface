def averaged(a):
    return sum(sum(a[:,2:4]))/(a.shape[0]*2)