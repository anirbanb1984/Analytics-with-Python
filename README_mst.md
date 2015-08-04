Exercise 2: Minimal Spanning Tree
For this problem we ask you to build a function that will compute a minimum spanning tree (MST) for a
set of n two-dimensional points in Euclidean space.
API
Your function should take a single parameter: a numpy array of size n x 2, where each row consists of the x,y
location of one point. The output will be a two-element tuple, where the first element is a single number
consisting of the length of the MST for the input set of points, and the second element is the actual tree. This
tree is represented as a list of n-1 two-element tuples, where each tuple represents a segment in the tree and
consists of two integers between 0 and n-1 representing which two rows of the input array are the two end
points of the segment.
Background
A spanning tree of a set of points P is a set of straight segments S with the properties that (1) It is possible to
travel from any point in P to any point in P through the segments S and (2) both end points of each segment in S
are points in the set P. The length of a spanning tree is the sum of the lengths of all its segments. A minimal
spanning tree is a spanning tree such that no other spanning tree has shorter length. We remark that minimal
spanning trees are not unique. For example any three sides of a square constitute an MST for four points lying in
the vertexes of the square. As seen in class, Kruskal's algorithm is a greedy algorithm that allows finding an
MST in V2log(V) time. (http://en.wikipedia.org/wiki/Kruskal%27s_algorithm).
