import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import list_files
from torchvision import datasets, models, transforms
from tqdm import tqdm, trange


class BasicClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(embedding_dim, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.softmax(x)


class BasicAutoencoder(nn.Module):
    """
    Taken from https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html
    """

    def __init__(self, in_shape, encoder_dim: int = 12, skip_decoding: bool = False):
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
        self.samples = list_files(root, suffix)
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)


def train_autoencoder(model: nn.Module, dataloader: DataLoader, num_epochs=25, lr=0.01):
    tqdm.write("Training Autoencoder...")

    criterion = nn.MSELoss()  # MSE loss for images
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in trange(num_epochs, desc="Epoch"):
        for inputs in dataloader:
            optimizer.zero_grad()
            decoded = model(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()


def train_classifier(model: nn.Module, dataloader: DataLoader, num_epochs=25, lr=0.01):
    tqdm.write("Training Classifier...")

    criterion = nn.BCELoss()  # Binary loss for classes
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in trange(num_epochs, desc="Epoch"):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()


INPUT_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create training and validation datasets
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}

ComicDataset()

model = BasicAutoencoder(INPUT_SIZE)
train_autoencoder()

model.skip_decoding = True
model2 = BasicClassifier(model)

train_classifier()
