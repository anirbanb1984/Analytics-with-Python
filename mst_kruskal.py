"""
An implementation of Minimum Spanning Tree functionality by Kruskal's algorithm
"""
from collections import OrderedDict
"""
Calculates the distance between 2 two-dimensional points in space
"""
def calculate_dist(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
"""
The mst_kruskal dunction takes an nX2 numpy array as a parameter
and returns a tuple whose first element is the distance traversed
along the MST and the second element of the tuple is a list of edges
represented in the MST 
"""
def mst_kruskal(array): 
    nodesList=[]
    mstList=[]
    edgeList=[]
    seen_edges=[]
    distances = {}
    color={}
    colornodes={}
    sumdist=0
    n=len(array)
    for i in range(1,n+1):
        colornodes[i]=[i-1]
    for i in range(n):
        nodesList.append(i)
        color[i]=i+1
    for i in range(n):
        for j in range(i+1,n):
            edgeList.append((i,j))
        
            
    for edge in edgeList:
        distances[edge]=calculate_dist(array[edge[0]],array[edge[1]])
    sorted_distances = OrderedDict(sorted(distances.items(),key=lambda x: x[1]))
    for edge in sorted_distances.keys():
        if edge not in seen_edges and color[edge[0]]!=color[edge[1]]:
            seen_edges.append(edge)
            temp=colornodes[color[edge[0]]]
            temp_copy=temp[:]
                
            for node in temp_copy:
                temp_col = color[node]
                color[node]=color[edge[1]]
                colornodes[temp_col].remove(node)
                colornodes[color[edge[1]]].append(node)
                
            mstList.append(edge)
            sumdist+=sorted_distances[edge]
    return (sumdist, mstList)
            
            
            
            
     
            
        
    
        
        
