# --> General imports
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch
import random
import pickle

# --> Torch imports 
import torch


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def set_seed(SEED, disable_cudnn=False):
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)        # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True  
    else:
        torch.backends.cudnn.enabled = False 


def collate_tangle(batch):
    feats, rna_data, patch_emb_aug, patch_emb_avg = zip(*batch)
    return torch.stack(feats, 0), torch.stack(rna_data, 0), torch.stack(patch_emb_aug, 0), torch.stack(patch_emb_avg, 0)


def collate_slide(batch):
    """
    Args:
        batch (List[Tuple[torch.Tensor, int]]): List of individual data points from the dataset.
    Returns:
        features_batch (torch.Tensor): Batch of feature tensors.
        labels_batch (torch.Tensor): Batch of labels.
    """
    features_list, ids_list = zip(*batch)
    features_batch = torch.stack(features_list, dim=0)
    return features_batch, ids_list


def smooth_rank_measure(embedding_matrix, eps=1e-7):
    """
    Compute the smooth rank measure of a matrix of embeddings.
    
    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension
        alpha (float): Smoothing parameter to avoid division by zero.

    Returns:
        float: Smooth rank measure.
    """
    
    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)
    
    # Compute the smooth rank measure
    p = S / torch.norm(S, p=1) + eps
    p = p[:embedding_matrix.shape[1]]
    smooth_rank = torch.exp(-torch.sum(p * torch.log(p)))
    smooth_rank = round(smooth_rank.item(), 2)
    
    return smooth_rank

