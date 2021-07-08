# -*- coding: utf-8 -*-
import numpy as np
###################################################################################################
#    define matrix

matrix = np.array([[1,2,3],[2,3,4]])  #列表转化为矩阵
print(matrix)
print('number of dim:',matrix.ndim)  # 维度
# number of dim: 2
print('shape :',matrix.shape)    # 行数和列数
# shape : (2, 3)
print('size:',matrix.size)   # 元素个数
# size: 6

a = np.array([2,23,4],dtype=np.int)
print(a.dtype)
# int 64
###################################################################################################
#    matrix operation

a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
b=np.arange(4)              # array([0, 1, 2, 3])
c=a-b  # array([10, 19, 28, 37])
c=a+b   # array([10, 21, 32, 43])
c=a*b   # array([  0,  20,  60, 120])
c=b**2  # array([0, 1, 4, 9])

c=10*np.sin(a)  
# array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])
print(b<3)  
# array([ True,  True,  True, False], dtype=bool)

print(a)
#[10 20 30 40]
print(a[b<3]) 
# 多条件： a[ (a>5) & ( a%2==0)]

#numpy 操作

#在某区间内生成某范围的整数序列
f = np.random.randint(low=0 , high=10, size=(10), dtype=np.int64)

#在矩阵中随机选点 并给每个店随机取值
import random
image = np.zeros((50,50),dtype = np.int)
x,y = np.where(image <=1)
numbers = range(0,len(x))
selected_pos = random.sample(numbers , len(x)//20 )
# 多维索引 每一位要单独构建一个list
for i in range(0 , len(selected_pos) ):
    image[ x[ selected_pos[i] ] ,y[ selected_pos[i] ] ] = random.randint(0,256)
plt.imshow(image)

#求一个list里元素的平均值
y,x = np.where(object1 ==colors[1])
center1_x = int(np.around( np.average(x) ))
center1_y = int(np.around( np.average(y) ))

#derivative of the 1D array
derivative = np.diff(x)
#长度减一

#[10 20 30]
a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape((2,2))
#arange(n) : 0 ~ n-1
print(a)
# array([[1, 1],
#       [0, 1]])
print(b)
# array([[0, 1],
#       [2, 3]])

c_dot = np.dot(a,b) #matrix multiply
# array([[2, 4],
#       [2, 3]])
c_dot_2 = a.dot(b)
# array([[2, 4],
#       [2, 3]])

###################################################################################################
#    operation on elements

np.random.uniform(low=0.0, high=1.0, size=None)
# 生出size个 符合均匀分布 的浮点数，取值范围为[low, high)，默认取值范围为[0, 1.0)

#np.random.rand(d0, d1, ..., dn)
#生成一个(d0, d1, ..., dn)维的数组，数组的元素取自[0, 1)上的均分布，若没有参数输入，则生成一个数
np.random.rand(3,2,1)
'''
array([[[0.00404447],
        [0.3837963 ]],

       [[0.32518355],
        [0.82482599]],

       [[0.79603205],
        [0.19087375]]])
'''

#numpy.random.randint(low, high=None, size=None, dtype='I')
#生成size个整数，取值区间为[low, high)，若没有输入参数high则取值区间为[0, low)
np.random.randint(12, size=(2,2,3), dtype=np.int64)

'''
2x2x3 [0-7]
array([[[5, 5, 6],
        [2, 7, 2]],

       [[2, 7, 6],
        [4, 7, 7]]], dtype=int64)
'''

#np.random.choice(a, size=None, replace=True, p=None)
#从a（数组）中选取size（维度）大小的随机数，replace=True表示可重复抽取，p是a中每个数出现的概率
#若a是整数，则a代表的数组是arange(a)
'''
def select( population , fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]
'''

#np.random.permutation(x)
# 随机打乱x中的元素。若x是整数，则打乱arange(x)，若x是一个数组，则将copy(x)的第一位索引打乱，
#意思是先复制x，对副本进行打乱处理，打乱只针对数组的第一维
np.random.permutation(np.arange(5))

#array([1, 2, 3, 0, 4])
np.random.permutation(np.array([[1,2,3],[4,5,6]]))
#array([[4, 5, 6],
#       [1, 2, 3]])

a=np.random.shuffle(np.arange(12))
#与permutation类似，随机打乱x中的元素。若x是整数，则打乱arange(x). 但是shuffle会对x进行修改

a=np.arange(12).reshape(2,2,3)
np.sum(a)   
np.min(a)   
np.max(a)   
np.argmin(a)    # 0
np.argmax(a)    # 11
np.mean(a)        # 7.5
np.median(a)       # 7.5

A = np.arange(14,2, -1).reshape((3,4))
# array([[14, 13, 12, 11],
#       [10,  9,  8,  7],
#       [ 6,  5,  4,  3]])
print(np.sort(A))    
# array([[11,12,13,14]
#        [ 7, 8, 9,10]
#        [ 3, 4, 5, 6]]) 
print(A.T)
# array([[14,10, 6]
#        [13, 9, 5]
#        [12, 8, 4]
#        [11, 7, 3]])
print(np.clip(A,5,9))    
# A中小于等于5的为变5 大于等于9的变为9 其他的不变
# array([[ 9, 9, 9, 9]
#        [ 9, 9, 8, 7]
#        [ 6, 5, 5, 5]])

###################################################################################################
#   np进行索引 
A = np.arange(3,15).reshape((3,4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
         
print(A[2])         
# [11 12 13 14]
print(A[1][1])      # 8
print(A[1, 1])      # 8
print(A[1, 1:3])    # [8 9] 
for row in A:
    print(row)
"""    
[ 3,  4,  5, 6]
[ 7,  8,  9, 10]
[11, 12, 13, 14]
"""
for column in A.T:
    print(column)
"""  
[ 3,  7,  11]
[ 4,  8,  12]
[ 5,  9,  13]
[ 6, 10,  14]
"""
print(A.flatten())   
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

for item in A.flat:
    print(item)
    
# 3
# 4
# ……
# 14
    
###################################################################################################
#    合并
A = np.array([1,1,1])
B = np.array([2,2,2])
         
print(np.vstack((A,B)))    # vertical stack
"""
[[1,1,1]
 [2,2,2]]
"""

D = np.hstack((A,B))       # horizontal stack

print(D)
# [1,1,1,2,2,2]

print(A.shape,D.shape)
# (3,) (6,)

A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]
D = np.concatenate((A,B,B,A),axis=1)

print(D)
"""
array([[1, 2, 2, 1],
       [1, 2, 2, 1],
       [1, 2, 2, 1]])
"""
###################################################################################################
#    zip x相当于捆绑 
a=[1,2,3,4,5]
b=(1,2,3,4,5)
c=np.arange(5)
d="zhang"
zz=zip(a,b,c,d)
for aa,bb,cc,dd in zz:
    print(aa,bb,cc,dd)

