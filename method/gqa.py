import torch 
import torch.nn as nn

# # （1）nn.Dropout随机丢弃,p就是丢弃的概率
# dropout_layer = nn.Dropout(p=0.5)

# t1 = torch.Tensor([1,2,3])
# t2 = dropout_layer(t1)
# #这里Dropout 为了保持期望不变，将其他部分扩大两倍
# print(t2)

# #（2）nn.Linear 线性层 对应用的张量乘以一个w矩阵然后+b
# layer = nn.Linear(in_features=3, out_features=5,bias = True)
# t1 = torch.Tensor([1,2,3]) #shape (3)
# t2 = torch.Tensor([[1,2,3]])#shape (1,3)
# output2 = layer(t1)
# #这里应用的w和b是随机的，真实训练里会在optimizer上更新
# print(output2)

# #（3）view 改变张量形状
# t = torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]])
# t_view1 = t.view(3,4)
# print(t_view1)
# t_view2 = t.view(2,3,2)
# print(t_view2)


# #（4）transpose 进行一个维度的交换
# t = torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]]) #[2,6]
# t = t.view(2,3,2)
# print(t)
# t = t.transpose(1,2)#交换1,2维度
# print(t)#[2,2,3]

#(5)triu 生成上三角矩阵
x = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# x = torch.triu(x) #生成上三角矩阵
# print(torch.triu(x))
# print(torch.triu(x,diagonal=1)) 
print(torch.full((3, 3),float("1")))
causal_mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
causal_mask=causal_mask.unsqueeze(0).unsqueeze(0)
print(causal_mask)
print(causal_mask.shape) #[1,1,3,3]

# #(6)reshape 改变张量形状
# x = torch.Tensor([1,2,3,4,5,6])
# y = torch.reshape(x,(2,3))
# print(y)

# #使用-1自动推断
# z = torch.reshape(x,(3,-1)) 
# print(z)

# # err = torch.reshape(x,(4,-1)) #不兼容的形状会报错
# # print(err)


# print(hasattr(torch.nn.functional, 'scaled_dot_product_attention'))#True 支持flashAttention