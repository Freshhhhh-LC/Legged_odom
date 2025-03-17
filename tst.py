import torch

# [24, 2, 2]
a = torch.randn(24, 1024, 50)
a = a[:,:,0]
print(a.shape)

# 