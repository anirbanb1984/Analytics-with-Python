"""
An implementation of einsum functionality in the numpy package
"""
"""
The combo function generates all possible combinations [r1,r2,...rk]
given a tuple (n1,n2,...nk) such that ri < ni for each i
"""
import numpy as np
def combo(n):
    r = [0]*len(n)
    while True:
        yield r
        p = len(r)-1
        r[p]+=1
        while r[p] == n[p]:
            r[p]=0
            p-=1
            if p == -1:
                return
            else:
                r[p]+=1

"""
The myeinsum function takes two parameters namely the subscripts which is
a string and operands which is a list of numpy arrays. The function returns
a numpy array as an output based on the operations specified in the subscripts
string.
"""
def myeinsum(subscripts,*operands):
    before_arrow = subscripts.split('->')[0]
    inputdims = before_arrow.split(',')
    finaldims = subscripts.split('->')[1]
    dict1={}
    if len(inputdims)!=len(operands):
        print 'Mismatch in dimensions'
    uniquedims = removedup(inputdims)
    for i in range(len(inputdims)):
        temp=inputdims[i]
        shape = np.shape(operands[i])
        for j in range(len(temp)):
            if temp[j] not in dict1.keys():
                dict1[temp[j]]=shape[j]
    finalshapes=()
    for i in range(len(uniquedims)):
        finalshapes+=(dict1[uniquedims[i]],)
    finalshapes1=()
    for i in range(len(finaldims)):
        finalshapes1+=(dict1[finaldims[i]],)
    out=np.zeros(finalshapes1)
    g = combo(finalshapes)
    final_tuple=()   
    for d in tuple(finaldims):
        final_tuple+=(finalshapes.index(dict1[d]),)
    list_initial_tuples = [() for i in range(len(operands))]
    for i in range(len(inputdims)):
        for d in tuple(inputdims[i]):
            list_initial_tuples[i]+=(finalshapes.index(dict1[d]),)
    for t in g: 
        prod=1
        for i in range(len(operands)):
            prod=prod*operands[i][tuple([t[j] for j in list_initial_tuples[i] ])]
        out[tuple([t[i] for i in final_tuple])]+=prod
    return out
        
"""
The removedup function takes a list of strings as an argument
and returns a tuple of unique strings after removing duplicate entries
"""    
def removedup(mylist):
    temp=''    
    for dim in mylist:
        temp+=dim
    result=''
    for c in temp:
        if c not in result:
            result+=c
    return tuple(result)
