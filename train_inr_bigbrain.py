import torch
import torch.nn.functional as F
import numpy as np
from siren import Siren, get_mgrid
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

x, _ = nib.load('../0BigBrain/rh.white.anat.surf.gii').agg_data()
y = nib.load('../0BigBrain/rh.DKTatlas40.label.gii').agg_data()

x -= x.mean(axis=0)
x *= (1 / np.abs(x).max()) * 0.999999

x = torch.from_numpy(x).to(torch.float32)
y = torch.from_numpy(y).to(torch.long)

for seed in range(2):

    print('seed: %d' % seed)

    torch.manual_seed(seed)

    writer = SummaryWriter(log_dir='runs/BigBrain/%d' % seed)

    model = Siren(device=device, in_features=3, out_features=36, hidden_features=512, hidden_layers=3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    model.train()
    losses = []
    accs = []
    for epoch in range(2000):
        x, y = x.to(model.device), y.to(model.device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=1)
        acc = (pred == y).float().mean()
        accs.append(acc.item())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss, global_step=epoch)
        writer.add_scalar("acc", acc, global_step=epoch)
        print(f'Epoch {epoch}, loss {loss.item()}, acc {acc.item()}')

    torch.save(model.state_dict(), '/home/daniel/data/BigBrain_INR/model_%d.pt' % seed)

    writer.close()

# /opt/conda/bin/python train_inr.py
