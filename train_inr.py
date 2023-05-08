import time
import torch
import torch.nn.functional as F
import numpy as np
from siren import Siren, get_mgrid
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

x = get_mgrid(28, 2)

# for i in range(10):

#     for seed in range(100):

#         print('label: %d, seed: %d' % (i, seed))

#         torch.manual_seed(seed)

#         writer = SummaryWriter(log_dir='runs/ProtoMNIST/%d_%d' % (i, seed))

#         y = np.load('/home/daniel/src/METALEARNING/MNIST/ProtoMNIST/%d.npy' % i)
#         y = torch.from_numpy(y).reshape(-1, 1)

#         model = Siren(device=device, in_features=2, out_features=1, hidden_features=512, hidden_layers=3)

#         optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

#         model.train()
#         losses = []
#         for epoch in range(2000):
#             x, y = x.to(model.device), y.to(model.device)
#             optimizer.zero_grad()
#             out = model(x)
#             loss = F.mse_loss(out, y)
#             losses.append(loss.item())
#             loss.backward()
#             optimizer.step()
#             writer.add_scalar("loss", loss, global_step=epoch)
#             print(f'Epoch {epoch}, loss {loss.item()}')

#         torch.save(model.state_dict(), '/home/daniel/data/ProtoMNIST/model_%d_%d.pt' % (i, seed))

#         writer.close()

# /opt/conda/bin/python train_inr.py


# full MNIST

import torchvision

train_set = torchvision.datasets.MNIST('/home/daniel/src/data/', train=True, download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))

test_set = torchvision.datasets.MNIST('/home/daniel/src/data/', train=False, download=False,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))

for i, data in enumerate(test_set):
    start = time.time()
    y, label = data
    x, y = x.to(device), y.reshape(-1, 1).to(device)
    writer = SummaryWriter(log_dir='runs/MNIST/%d_%d' % (i, label))
    model = Siren(device=device, in_features=2, out_features=1, hidden_features=512, hidden_layers=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    model.train()
    for epoch in range(2000):
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss, global_step=epoch)
        # print(f'Epoch {epoch}, loss {loss.item()}')
    torch.save(model.state_dict(), '/home/daniel/data/MNIST/test/model_%d_%d.pt' % (i, label))
    writer.close()
    stop = time.time()
    print('%d label: %d time: %f' % (i, label, stop - start))
