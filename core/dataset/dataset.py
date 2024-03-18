# --> General imports
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle

# --> Torch imports 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from tensorboardX import SummaryWriter


class TangleDataset(Dataset):
    def __init__(self, feats_dir, rna_dir, n_tokens, sampling_strategy="random"):
        """
        - feats_dir: str, directory where feat .pt files are stored
        - rna_dir: str, directory where rna_data .pt files are stored
        - n_tokens: int, number of tokens/patches to sample from feats
        - sampling_strategy: str, strategy to sample patches ("random" or "kmeans_cluster")
        """
        self.feats_dir = feats_dir
        self.rna_dir = rna_dir
        self.n_tokens = n_tokens
        self.sampling_strategy = sampling_strategy
        
        slide_ids = [fname.split(".pt")[0] for fname in os.listdir(feats_dir) if fname.endswith(".pt")]
        rna_ids = [fname.split(".pt")[0] for fname in os.listdir(rna_dir) if fname.endswith(".pt")]
        self.slide_ids = list(set(slide_ids) & set(rna_ids))

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        
        # Load features and coords 
        patch_emb = torch.load(os.path.join(self.feats_dir, f"{slide_id}.pt"))

        # - Avg patch embedding 
        patch_emb_avg = patch_emb.mean(dim=0)
        
        # Original 
        patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
        patch_emb_ = patch_emb[patch_indices]

        # And an augmentation
        patch_indices_aug = torch.randint(0, patch_emb.size(0), (self.n_tokens,)).tolist() if patch_emb.shape[0] < self.n_tokens else torch.randperm(patch_emb.size(0))[:self.n_tokens].tolist()           
        patch_emb_aug = patch_emb[patch_indices_aug]

        # Load gene expression data 
        rna_data = torch.load(os.path.join(self.rna_dir, f"{slide_id}.pt"))

        return patch_emb_, rna_data, patch_emb_aug, patch_emb_avg


class SlideDataset(Dataset):
    def __init__(self, csv_path, features_path):
        """
        Args:
            csv_path (string): Path to the csv file with labels and slide_id.
            features_path (string): Directory with all the feature files.
        """
        self.dataframe = pd.read_csv(csv_path)
        self.features_path = features_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        slide_id = self.dataframe.iloc[index, self.dataframe.columns.get_loc('slide_id')]
        try:
            feature_file = f"{slide_id}.pt"
            feature_path = f"{self.features_path}/{feature_file}"
            features = torch.load(feature_path)
            label = self.dataframe.iloc[index, self.dataframe.columns.get_loc('label')]
        except:
            print(slide_id, ' not found!! Return dummy values.')
            features = torch.Tensor([-1])
            label = torch.Tensor([-1])

        return features, label
