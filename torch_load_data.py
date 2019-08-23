import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import numpy as np

from kde_training_utilities import kde_load_data

class KdeDataset(Dataset):
    def __init__(self, folder="/users/afengler/data/kde/ddm/train_test_data/", 
        log=True, cutoff=1e-7):
        X, y, X_val, y_val = kde_load_data(
                folder, log=log, prelog_cutoff=cutoff)
        self.X = torch.tensor(np.array(X), dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.X_val = torch.tensor(np.array(X_val), dtype=torch.float)
        self.y_val = torch.tensor(y_val, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KdeFCNDataset(Dataset):
    def __init__(self, folder):
        X = pickle.load(open(folder + "/train_features.pickle", "rb"))
        y = pickle.load(open(folder + "/train_labels.pickle", "rb"))
        X_val = pickle.load(open(folder + "/test_features.pickle", "rb"))
        y_val = pickle.load(open(folder + "/test_labels.pickle", "rb"))

        self.X = torch.tensor(np.swapaxes(X, 1, 2), dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.X_val = torch.tensor(np.swapaxes(X_val, 1, 2), dtype=torch.float)
        self.y_val = torch.tensor(y_val, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KdeFCNArrayDataset(Dataset):
    def __init__(self, folder="/users/afengler/data/kde/ddm/train_test_data/", 
        log=True, cutoff=1e-7, array_size=1000):
        X, y, X_val, y_val = kde_load_data(
                folder, log=log, prelog_cutoff=cutoff)
        self.X = torch.tensor(np.array(X).T, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.X_val = torch.tensor(np.array(X_val).T, dtype=torch.float)
        self.y_val = torch.tensor(y_val, dtype=torch.float)
        self.array_size = array_size

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        temp_idx = torch.randint(high=self.X.shape[1], size=(self.array_size,))
        return self.X[:, temp_idx], self.y[temp_idx].squeeze()

