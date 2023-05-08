import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import nibabel as nib

def train(args, model, device, train_loader, optimizer, epoch, softmax=False):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if softmax:
            output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            acc = 100. * correct / len(train_loader.dataset)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    acc = 100. * correct / len(train_loader.dataset)
    print('Train Accuracy: ({:.0f}%) '.format(acc))


def test(model, device, x, y, id):
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         if softmax:
    #             output = F.log_softmax(output, dim=1)
    #         test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    # acc = 100. * correct / len(test_loader.dataset)
    # print('\nAverage loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
    #     test_loss, acc))
    # return test_loss, acc
    
    model.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        output = model(x)

        # pred = output.argmax(dim=1).cpu().numpy()
        # img = nib.load('../0BigBrain/rh.DKTatlas40.label.gii')
        # img_data = img.agg_data()
        # img.remove_gifti_data_array(0)
        # img.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pred, intent='NIFTI_INTENT_LABEL', datatype='NIFTI_TYPE_INT32'))
        # nib.save(img, f'animation/pred{id}.label.gii')
        # test_loss = F.cross_entropy(output, y).item()
        
        plt.imshow(output.cpu().numpy().reshape(28, 28))
        plt.savefig(f"animation/output_{id}.png", dpi=100)
        test_loss = F.mse_loss(output, y).item()
    print('\nAverage loss: {:.4f}\n'.format(test_loss))
    return test_loss
