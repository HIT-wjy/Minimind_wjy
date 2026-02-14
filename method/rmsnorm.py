import torch


#开方求倒数
t1 = torch.rsqrt(torch.tensor([1,4,25]))
print(t1)

#创建一个全1张量
t2 = torch.ones(3,2)
print(t2)

x = torch.tensor([[1,2,3,4,5],
                  [6,7,8,9,10]])
x = x.float()
y = x.pow(2).mean(-1)
print(y)