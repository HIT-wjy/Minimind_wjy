import torch


#1. torch.where
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

condition = x>3

result = torch.where(condition,x,y)

# print(result)


#2. torch.arange
t = torch.arange(0,10,2)
# print(t)

t1 = torch.arange(5,0,-2)
# print(t1)

#3. torch.outer
v1 = torch.tensor([1,2,3])
v2 = torch.tensor([4,5,6])
result = torch.outer(v1,v2)
# print(result)

#4. torch.cat
t1 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
t2 = torch.tensor([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
# print(t1.shape) # [2, 2, 3]

# result = torch.cat([t1,t2],dim=0)
# print(result)
# print(result.shape) # [4, 2, 3]

# result2 = torch.cat([t1,t2],dim=1)
# print(result2)
# print(result2.shape) # [2, 4, 3]

#5. unsqueeze
t = torch.tensor([1,2,3])
print(t.shape)
result = t.unsqueeze(0)
print(result)
print(result.shape)
