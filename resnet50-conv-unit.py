import torch
import time

device = torch.device("cuda:0")  # Uncomment this to run on GPU

# self.conv1
conv1 = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False).cuda()

input = torch.randn((32, 64, 56, 56), device=device)

t = time.time()

with torch.no_grad():
    output = conv1(input)

print(output[0][0][0][0])
print('Elapsed time: {:.4f}s'.format(time.time() - t))