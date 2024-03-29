from models.mlp import MLP
from siren import Siren, get_mgrid
from utils.weight_matching import mlp_permutation_spec, weight_matching, apply_permutation
from utils.utils import flatten_params, lerp
from utils.plot import plot_interp_acc
import argparse
import torch
from torchvision import datasets, transforms
from utils.training import test
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import nibabel as nib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # load models
    model_a = Siren(device=torch.device('cpu'), in_features=2, hidden_features=512, out_features=1, hidden_layers=3)
    model_b = Siren(device=torch.device('cpu'), in_features=2, hidden_features=512, out_features=1, hidden_layers=3)
    checkpoint = torch.load(args.model_a)
    model_a.load_state_dict(checkpoint)
    checkpoint_b = torch.load(args.model_b)
    model_b.load_state_dict(checkpoint_b)

    permutation_spec = mlp_permutation_spec(4)
    final_permutation = weight_matching(permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))
              

    updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b))

    
    # # test against mnist
    # transform=transforms.Compose([
    #   transforms.ToTensor(),
    #   transforms.Normalize((0.1307,), (0.3081,))
    #   ])
    # test_kwargs = {'batch_size': 5000}
    # train_kwargs = {'batch_size': 5000}
    # dataset = datasets.MNIST('../data', train=False,
    #                   transform=transform)
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                   transform=transform)

    x = get_mgrid(28, 2)
    y = np.load('/home/daniel/src/METALEARNING/MNIST/ProtoMNIST/0.npy')
    y = torch.from_numpy(y).reshape(-1, 1)

    import torchvision
    test_set = torchvision.datasets.MNIST('/home/daniel/src/data/', train=False, download=False,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    y = test_set[3][0].reshape(-1, 1)

    # x, _ = nib.load('../0BigBrain/rh.white.anat.surf.gii').agg_data()
    # y = nib.load('../0BigBrain/rh.DKTatlas40.label.gii').agg_data()
    # x -= x.mean(axis=0)
    # x *= (1 / np.abs(x).max()) * 0.999999
    # x = torch.from_numpy(x).to(torch.float32)
    # y = torch.from_numpy(y).to(torch.long)

    lambdas = torch.linspace(0, 1, steps=25)

    test_acc_interp_clever = []
    test_acc_interp_naive = []
    train_acc_interp_clever = []
    train_acc_interp_naive = []
    # naive
    model_b.load_state_dict(checkpoint_b)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for iter, lam in enumerate(tqdm(lambdas)):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      acc = test(model_b.cuda(), 'cuda', x, y, str(iter) + '_naive')
      test_acc_interp_naive.append(acc)
      # acc = test(model_b.cuda(), 'cuda', x, y)
      # train_acc_interp_naive.append(acc)

    # smart
    model_b.load_state_dict(updated_params)
    model_b.cuda()
    model_a.cuda()
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for iter, lam in enumerate(tqdm(lambdas)):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      acc = test(model_b.cuda(), 'cuda', x, y, str(iter) + '_clever')
      test_acc_interp_clever.append(acc)
      # acc = test(model_b.cuda(), 'cuda', x, y)
      # train_acc_interp_clever.append(acc)

    fig = plot_interp_acc(lambdas, test_acc_interp_naive, test_acc_interp_naive,
                    test_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_mlp_weight_matching_interp_accuracy_epoch.png", dpi=200)

    # os.system('zip -r animation.zip animation')
    # os.system('rm animation/*.gii')

    naives = []
    clevers = []
    for iter in range(len(lambdas)):
      image = imageio.v2.imread(f'animation/output_{str(iter)}_naive.png')
      naives.append(image)
      image = imageio.v2.imread(f'animation/output_{str(iter)}_clever.png')
      clevers.append(image)
    imageio.mimsave('animation/output_naive.gif', naives, fps = 2)
    imageio.mimsave('animation/output_clever.gif', clevers, fps = 2)
    os.system('rm animation/*.png')

if __name__ == "__main__":
  main()


# /opt/conda/bin/python mlp_mnist_weight_matching.py --model_a model_0.pt --model_b model_1.pt

# /opt/conda/bin/python mlp_mnist_weight_matching.py --model_a /home/daniel/data/ProtoMNIST/model_0_0.pt --model_b /home/daniel/data/ProtoMNIST/model_0_1.pt
