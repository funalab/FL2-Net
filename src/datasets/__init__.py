from src.datasets.dataset import PreprocessedDataset
from torch.utils.data import DataLoader


def get_dataloader(cfg, train=True):
    
    train_dataset = PreprocessedDataset(cfg, train)
    return train_iterator
