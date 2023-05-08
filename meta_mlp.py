import os
import copy
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from siren import Siren, get_mgrid

from utils.weight_matching import mlp_permutation_spec, weight_matching, apply_permutation
from utils.utils import flatten_params, lerp
from utils.plot import plot_interp_acc

path = '/home/daniel/data/ProtoMNIST'

hidden = 10

if False:

    references = [None for _ in range(10)]
    models = [[] for _ in range(10)]

    for n, file in enumerate(sorted(os.listdir(path))):
        name, ext = os.path.splitext(file)
        if ext == '.pt':
            y = torch.tensor((int(name.split('_')[-2]),), dtype=torch.int64)
            model = Siren(device=torch.device('cpu'), in_features=2, out_features=1,
                        hidden_features=512, hidden_layers=3)
            model.load_state_dict(torch.load(os.path.join(path, file)))
            model.eval()
            if references[y.item()] is None:
                references[y.item()] = copy.deepcopy(model) # reference models
            models[y.item()].append(copy.deepcopy(model))
            print(n)

    aligned_dataset = []
    unaligned_dataset = []

    def test_align(model, ref, y, lambdas):
        loss_interp = []
        ref.eval()
        ref_dict = copy.deepcopy(ref.state_dict())
        model.eval()
        model_dict = copy.deepcopy(model.state_dict())
        x = get_mgrid(28, 2)
        for lam in lambdas:
            p = lerp(lam, ref_dict, model_dict)
            model.load_state_dict(p)
            with torch.no_grad():
                output = model(x)
                img = np.load('/home/daniel/src/METALEARNING/MNIST/ProtoMNIST/%d.npy' % y)
                img = torch.from_numpy(img).reshape(-1, 1)
                loss = F.mse_loss(output, img).item()
            loss_interp.append(loss)
        return loss_interp

    lambdas = torch.linspace(0, 1, steps=25)
    for i, model_list in enumerate(models):
        ref = copy.deepcopy(references[i])
        for j, model in enumerate(model_list):
            loss_interp_naive = test_align(copy.deepcopy(model), copy.deepcopy(ref), i, lambdas)
            parameters = torch.cat([p.flatten().detach() for p in model.parameters() if p.requires_grad])[None, :]
            parameters = parameters.detach().squeeze(0)
            unaligned_dataset.append((parameters, i))
            # alignment
            if j == 0:
                aligned_dataset.append((parameters, i))
            permutation_spec = mlp_permutation_spec(4)
            final_permutation = weight_matching(permutation_spec, flatten_params(ref), flatten_params(model))
            updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model))
            model.load_state_dict(updated_params)
            loss_interp_clever = test_align(copy.deepcopy(model), copy.deepcopy(ref), i, lambdas)
            parameters = torch.cat([p.flatten().detach() for p in model.parameters() if p.requires_grad])[None, :]
            parameters = parameters.detach().squeeze(0)
            aligned_dataset.append((parameters, i))
            # ready
            print(i, j)
            fig = plot_interp_acc(lambdas,
                                  loss_interp_naive, loss_interp_naive,
                                  loss_interp_clever, loss_interp_clever)
            plt.savefig(f"animation/mnist_mlp_weight_matching_interp_loss_{i}_{j}.png", dpi=200)
            plt.close()

    # save aligned models
    for x, y in aligned_dataset:
        folder = '/home/daniel/data/ProtoMNIST_parameters/aligned'
        count = len([name for name in os.listdir(folder) \
                    if os.path.isfile(os.path.join(folder, name)) and name.startswith(str(y))])
        np.save(os.path.join(folder, f"{y}_{count}.npy"), x.numpy())

    # save unaligned models
    for x, y in unaligned_dataset:
        folder = '/home/daniel/data/ProtoMNIST_parameters/unaligned'
        count = len([name for name in os.listdir(folder) \
                    if os.path.isfile(os.path.join(folder, name)) and name.startswith(str(y))])
        np.save(os.path.join(folder, f"{y}_{count}.npy"), x.numpy())

else:
    aligned_dataset = []
    ppath = '/home/daniel/data/ProtoMNIST_parameters/aligned'
    for file in sorted(os.listdir(ppath)):
        if file.endswith('.npy'):
            param = np.load(os.path.join(ppath, file))
            param = torch.from_numpy(param)
            y = torch.tensor((int(file.split('_')[0]),), dtype=torch.int64)
            aligned_dataset.append((param, y))


np.random.shuffle(aligned_dataset)

train_loader = DataLoader(aligned_dataset[:800], batch_size=20, shuffle=True)
val_loader = DataLoader(aligned_dataset[800:900], batch_size=20)
test_loader = DataLoader(aligned_dataset[900:], batch_size=20)

class MLP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lin1 = nn.Linear(790017, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, 10)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        return x

metamodel = MLP(device)

print(metamodel)

metaoptimizer = torch.optim.AdamW(metamodel.parameters(), lr=0.0001, weight_decay=0.001)

def train(loader, model, optimizer):
    model.train()
    losses = []
    cms = []
    for x, y in loader:
        x, y = x.to(model.device), y.to(model.device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y.squeeze(-1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        cm = confusion_matrix(y.detach().cpu().numpy(),
                              pred.detach().cpu().numpy(),
                              labels=np.arange(10))
        cms.append(cm)
    return sum(losses) / len(losses), sum(cms)

def test(loader, model):
    model.eval()
    losses = []
    cms = []
    for x, y in loader:
        x, y = x.to(model.device), y.to(model.device)
        out = model(x)
        loss = F.cross_entropy(out, y.squeeze(-1))
        losses.append(loss.item())
        pred = out.argmax(dim=1)
        cm = confusion_matrix(y.detach().cpu().numpy(),
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
for epoch in range(100):
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
plt.savefig('meta_mlp_loss.png')
plt.clf()
plt.plot(train_accs)
plt.plot(val_accs)
plt.savefig('meta_mlp_acc.png')
plt.clf()

loss, cm = test(test_loader, metamodel)
ConfusionMatrixDisplay(cm).plot()
plt.savefig('meta_mlp_cm.png')
print(get_acc(cm))
