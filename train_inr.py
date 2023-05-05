import torch
import torch.nn.functional as F
import numpy as np
from siren import Siren, get_mgrid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 1

torch.manual_seed(seed)

y = np.load('/home/daniel/src/METALEARNING/MNIST/ProtoMNIST/1.npy')
y = torch.from_numpy(y).reshape(-1, 1)

x = get_mgrid(28, 2)

model = Siren(device=device, in_features=2, out_features=1, hidden_features=64, hidden_layers=3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
losses = []
for epoch in range(1000):
    x, y = x.to(model.device), y.to(model.device)
    optimizer.zero_grad()
    out = model(x)
    loss = F.mse_loss(out, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, loss {loss.item()}')

torch.save(model.state_dict(), 'model_%d.pt' % seed)

# /opt/conda/bin/python train_inr.py
