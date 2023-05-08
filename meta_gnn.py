import os
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from Model2Graph import getEdgesQuickly
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import torch_geometric.nn as pyg_nn
from GNLayer import IntroGNLayer, GNLayer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from siren import Siren, get_mgrid

path = '/home/daniel/data/ProtoMNIST'

dataset = []

hidden = 10

for n, file in enumerate(os.listdir(path)):
    name, ext = os.path.splitext(file)
    if ext == '.pt':
        y = torch.tensor((int(name.split('_')[-2]),), dtype=torch.int64)
        model = Siren(device=torch.device('cpu'), in_features=2, out_features=1, hidden_features=512, hidden_layers=3)
        model.load_state_dict(torch.load(os.path.join(path, file)))
        model.eval()
        edge_index, edge_weight, num_nodes = getEdgesQuickly(model)
        edge_index, edge_weight = edge_index.detach(), edge_weight.detach()
        # edge_weight += torch.normal(0.0, 0.02, size=edge_weight.shape)
        edge_index, edge_weight = to_undirected(edge_index, edge_weight)
        dataset.append(Data(x=None,
                            edge_index=edge_index,
                            edge_weight=edge_weight.unsqueeze(-1),
                            y=y,
                            num_nodes=num_nodes))
        print(n)

np.random.shuffle(dataset)

train_loader = DataLoader(dataset[:800], batch_size=5, shuffle=True)
val_loader = DataLoader(dataset[800:900], batch_size=5)
test_loader = DataLoader(dataset[900:], batch_size=5)

class GNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv1 = IntroGNLayer(hidden, hidden, 1, act_fn=nn.ReLU())
        self.conv2 = GNLayer(hidden, hidden, hidden, 1, act_fn=nn.ReLU())
        self.lin3 = nn.Linear(2*hidden, hidden)
        self.lin4 = nn.Linear(hidden, 10)
        self.device = device
        self.to(self.device)

    def forward(self, data):
        x = data.x
        edge_index, edge_weight = data.edge_index, data.edge_weight
        batch = data.batch
        x = self.conv1(edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = torch.cat([pyg_nn.global_mean_pool(x, batch),
                       pyg_nn.global_max_pool(x, batch)], dim=1)
        x = self.lin3(x)
        x = torch.relu(x)
        x = self.lin4(x)
        return x

metamodel = GNN(device)

print(metamodel)

metaoptimizer = torch.optim.Adam(metamodel.parameters(), lr=0.001)

def train(loader, model, optimizer):
    model.train()
    losses = []
    cms = []
    for data in loader:
        data = data.to(model.device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        cm = confusion_matrix(data.y.detach().cpu().numpy(),
                              pred.detach().cpu().numpy(),
                              labels=np.arange(10))
        cms.append(cm)
    return sum(losses) / len(losses), sum(cms)

def test(loader, model):
    model.eval()
    losses = []
    cms = []
    for data in loader:
        data = data.to(model.device)
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        losses.append(loss.item())
        pred = out.argmax(dim=1)
        cm = confusion_matrix(data.y.detach().cpu().numpy(),
                              pred.detach().cpu().numpy(),
                              labels=np.arange(10))
        cms.append(cm)
    return sum(losses) / len(losses), sum(cms)

def get_acc(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
for epoch in range(1000):
    loss, cm = train(train_loader, metamodel, metaoptimizer)
    train_losses.append(loss)
    train_accs.append(get_acc(cm))
    loss, cm = test(val_loader, metamodel)
    val_losses.append(loss)
    val_accs.append(get_acc(cm))
    print(epoch, train_losses[-1], train_accs[-1], val_losses[-1], val_accs[-1])

plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.legend()
plt.savefig('meta_gnn_loss.png')
plt.clf()
plt.plot(train_accs)
plt.plot(val_accs)
plt.savefig('meta_gnn_acc.png')
plt.clf()

loss, cm = test(test_loader, metamodel)
ConfusionMatrixDisplay(cm).plot()
plt.savefig('meta_gnn_cm.png')
print(get_acc(cm))
