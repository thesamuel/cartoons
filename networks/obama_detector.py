import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms, models
import copy


class BasicConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(29 * 29 * 128, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def data_loaders(data_dir: str, use_cuda: bool, batch_size: int, val_batch_size: int):
    data_dir = Path(data_dir)
    input_size = 224
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(
        datasets.ImageFolder(data_dir / "train",
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(input_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        datasets.ImageFolder(data_dir / "val",
                             transform=transforms.Compose([
                                 transforms.Resize(input_size),
                                 transforms.CenterCrop(input_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])),
        batch_size=val_batch_size, shuffle=True, **kwargs
    )

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * test_accuracy:.0f}%)\n')

    return test_accuracy


def train_helper(seed: int, data_dir: str, use_cuda: bool, batch_size: int, val_batch_size: int, epochs: int,
                 log_interval: int, lr: float, momentum: float):
    # Setup torch
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get data loaders
    train_loader, test_loader = data_loaders(data_dir, use_cuda, batch_size, val_batch_size)

    # Create classifier
    model = models.AlexNet(num_classes=100)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Run training loop
    best_accuracy = 0
    best_model = None
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
        epoch_accuracy = test(model, device, test_loader, criterion)

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model = copy.deepcopy(model.state_dict())

    return best_model


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='path to data directory')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    best_model = train_helper(args.seed, args.data_dir, use_cuda, args.batch_size, args.val_batch_size, args.epochs,
                              args.log_interval, args.lr, args.momentum)

    if args.save_model:
        torch.save(best_model,
                   f"basic-cnn-lr-{args.lr}-momentum-{args.momentum}-batch-size-{args.batch_size}-best.pth")


if __name__ == '__main__':
    main()
