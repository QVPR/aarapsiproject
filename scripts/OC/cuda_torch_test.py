#!/usr/bin/env python3
import torch
import torchvision

print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
print(torchvision.__version__)

torch.cuda.empty_cache()
device = torch.device('cuda')
torch.nn.functional.conv2d(torch.zeros(32, 32, 32, 32, device=device), torch.zeros(32, 32, 32, 32, device=device))

print("Success!")
