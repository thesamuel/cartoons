import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import list_files
from tqdm import tqdm, trange

######################################################################
# Inputs
######################################################################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 224
INPUT_SHAPE = (3, INPUT_SIZE, INPUT_SIZE)

BATCH_SIZE = 32
LEARNING_RATE = 0.01

AUTOENCODER_DATA_DIR = './data/clean-data'
NUM_EPOCHS_AUTOENCODER = 1
ENCODER_DIM = 12

CLASSIFIER_DATA_DIR = './data/classifier-data'
NUM_EPOCHS_CLASSIFIER = 1
NUM_CLASSES = 2


######################################################################
# Models
######################################################################

class BasicClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder_dim, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.softmax(x)


class BasicAutoencoder(nn.Module):
    """
    Taken from https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html
    """

    def __init__(self, in_shape, encoder_dim: int, skip_decoding: bool = False):
        super().__init__()
        self.skip_decoding = skip_decoding
        c, h, w = in_shape

        hidden_dim_1 = 128
        hidden_dim_2 = 64

        self.encoder = nn.Sequential(
            nn.Linear(c * h * w, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, encoder_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, c * h * w),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, h, w = x.size()
        x = x.view(bs, -1)
        x = self.encoder(x)

        if not self.skip_decoding:
            x = self.decoder(x)
            x = x.view(bs, c, h, w)

        return x


class ComicDataset(Dataset):
    def __init__(self, root: str, suffix=".jpg", transform=None):
        super(ComicDataset, self).__init__()
        self.loader = default_loader
        self.samples = list_files(root, suffix, prefix=True)
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)


######################################################################
# Helpers
######################################################################

def train_loop_classifier(model: BasicClassifier, dataloaders: dict, criterion: nn.Module, optimizer: Optimizer,
                          num_epochs: int):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for _ in trange(num_epochs, desc='epoch'):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Only track history training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Only perform backward + optimization in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Make a deep copy of the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_loop_autoencoder(model: BasicAutoencoder, dataloaders: dict, criterion: nn.Module, optimizer: Optimizer,
                           num_epochs: int):
    since = time.time()

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for _ in trange(num_epochs, desc='epoch'):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for inputs in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Only track history training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)

                    # Only perform backward + optimization in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            tqdm.write(f'{phase} Loss: {epoch_loss:.4f}')

            # Make a deep copy of the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    tqdm.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    tqdm.write(f'Best val loss: {best_loss:4f}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history


def train_autoencoder(data_transforms: dict):
    # Create entire dataset
    dataset = ComicDataset(AUTOENCODER_DATA_DIR)

    # Split dataset into train and val
    split = int(0.9 * len(dataset))
    train_dataset, val_dataset, _ = random_split(dataset, [
        10,
        20,
        len(dataset) - 30
        # split, 
        # len(dataset) - split
    ])
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    image_datasets = {'train': train_dataset, 'val': val_dataset}

    # Create training and validation dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    # Initialize autoencoder
    autoencoder = BasicAutoencoder(INPUT_SHAPE, ENCODER_DIM)
    autoencoder.to(DEVICE)

    criterion = nn.MSELoss()  # MSE loss for images
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    tqdm.write("Training Autoencoder...")
    return train_loop_autoencoder(autoencoder, dataloaders, criterion, optimizer, NUM_EPOCHS_AUTOENCODER)


def train_classifier(trained_autoencoder: BasicAutoencoder, data_transforms: dict):
    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(CLASSIFIER_DATA_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Create training and validation dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    # Disable decoding on the autoencoder so we can use its embeddings
    # TODO: split autoencoder forward into two methods
    trained_autoencoder.skip_decoding = True

    # Initialize classifier with trained autoencoder
    classifier = BasicClassifier(trained_autoencoder, ENCODER_DIM, NUM_CLASSES)
    classifier.to(DEVICE)

    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Binary loss for classes
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    tqdm.write("Training Classifier...")
    return train_loop_classifier(classifier, dataloaders, criterion, optimizer, NUM_EPOCHS_CLASSIFIER)


def plot(label: str, num_epochs: int, hist: list):
    # Plot the training curves of validation accuracy vs. number of training epochs
    plt.clf()
    plt.title("Validation Metric vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Metric")
    plt.plot(range(1, num_epochs + 1), hist, label=label)
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{label}.png")


######################################################################
# Run Training and Validation Step
######################################################################

# Training: data augmentation and normalization
# Validation: only normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Train and evaluate
autoencoder, autoencoder_hist = train_autoencoder(data_transforms)
torch.save(autoencoder, "autoencoder-best.pth")
plot("Autoencoder", NUM_EPOCHS_AUTOENCODER, autoencoder_hist)

classifier, classifier_hist = train_classifier(autoencoder, data_transforms)
classifier_hist = [h.cpu().numpy() for h in classifier_hist]
torch.save(classifier, "classifier-best.pth")
plot("Classifier", NUM_EPOCHS_CLASSIFIER, classifier_hist)

