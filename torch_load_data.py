import torch
from torch.util.data import Dataset, Dataloader

from kde_training_utilities import kde_load_data

class KdeDataset(Dataset):

    def __init__(folder="/users/afengler/data/kde/ddm/train_test_data/"):
        self.X, self.y, self.X_val, self.y_val = kde_load_data(
                folder, log=True, prelog_cutoff="none")
        # perm = np.arange(self.X.shape[0])
        # perm = np.random.shuffle
        # self.X = self.X[perm]
        # self.y = self.y[perm]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"feature":self.X[idx], "likelihood":self.y[idx]}
