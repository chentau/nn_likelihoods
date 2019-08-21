import torch
import pickle
from torch.utils.data import Dataset
import numpy as np

from kde_training_utilities import kde_load_data

class KdeDataset(Dataset):
    def __init__(self, folder="/users/afengler/data/kde/ddm/train_test_data/", cutoff=1e-7):
        X, y, X_val, y_val = kde_load_data(
                folder, log=True, prelog_cutoff=cutoff)
        self.X = torch.tensor(np.array(X), dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.X_val = torch.tensor(np.array(X_val), dtype=torch.float)
        self.y_val = torch.tensor(y_val, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KdeFCNDataset(Dataset):
    def __init__(folder):
        X = pickle.load(open(folder + "/train_features.pickle", "rb"))
        y = pickle.load(open(folder + "/train_labels.pickle", "rb"))
        X_val = pickle.load(open(folder + "/test_features.pickle", "rb"))
        y_val = pickle.load(open(folder + "/test_labels.pickle", "rb"))

        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.X_val = torch.tensor(X_val, dtype=torch.float)
        self.y_val = torch.tensor(y_val, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
