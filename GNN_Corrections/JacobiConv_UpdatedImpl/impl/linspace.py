import numpy as np
import torch

n = 20
gamma = 0.2
num = int(n*gamma)
a = np.linspace(-1, 0,num)
b = np.linspace(0, 1, n-num)

a = torch.FloatTensor(a)

b = torch.FloatTensor(b)


c = torch.cat((a,b),0)

print(a)
print(b)

print(c)
