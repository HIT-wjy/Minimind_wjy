from numpy import bincount, repeat
from sympy import print_fcode
import torch
# out = torch.zeros(5)
# index = torch.tensor([0,1,0,3])
# src = torch.tensor([1.,2.,3.,4.])
# out.scatter_add_(dim=0,index=index,src=src)
# print(out)

# a = torch.tensor([10,20,30])
# b = torch.tensor([1,2,3])
# c= torch.div(a,b)
# print(c)

# x = torch.tensor([10.0,20.0,30.0])
# x = torch.mean(x,dim=0)
# print(x)
# y = torch.tensor([[10.0,20.0,30.0],[2.0,4.0,5.0]])
# print("dim=0:",torch.mean(y,dim=0))
# print("dim=1:",torch.mean(y,dim=1))

x = torch.tensor([1,2,3])
y = torch.repeat_interleave(x,repeats=2)
print(y)#tensor([1, 1, 2, 2, 3, 3])

#argsort:返回排序后元素在原数组的索引
x = torch.tensor([30,10,20])
idx = torch.argsort(x)
print(idx)
print(x[idx])

#bincount :统计非负整数出现的次数
x = torch.tensor([30,10,20])
num = bincount(x)
print(num)