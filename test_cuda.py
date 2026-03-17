import torch
from typing import List
from collections import defaultdict
# print("CUDA available:", torch.cuda.is_available())
# class Solution:
#     def twosum(self,nums:int,target:int)->List[int]:
#         cnt = defaultdict()
#         for i,x in enumerate(nums):
#             com = target-x
#             if com in cnt:
#                 return [i,cnt[com]]
#             cnt[x]=i
# if __name__ == "__main__":
#     s = Solution()
#     nums = [2,7,8,5]
#     target = 10
#     ans = s.twosum(nums,target)
#     print(ans)

s = "cbr"
print(sorted(s))
print("".join(sorted(s)))