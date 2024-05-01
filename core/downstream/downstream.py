import os
from tqdm import tqdm 
import numpy as np

import torch 
from torch.utils.data import DataLoader

from core.dataset.dataset import SlideDataset
from core.utils.learning import collate_slide, save_pkl, smooth_rank_measure


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_loop(ssl_model, val_dataloader):
    
    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)
    
    all_embeds = []
    all_slide_ids = []
    
    # do everything without grads 
    with torch.no_grad():
        for inputs, slide_id in tqdm(val_dataloader):
            inputs = inputs.to(DEVICE)
            wsi_embed = ssl_model.get_features(inputs)
            wsi_embed = wsi_embed.float().detach().cpu().numpy()
            all_embeds.extend(wsi_embed)
            all_slide_ids.extend(slide_id)
            
    all_embeds = np.array(all_embeds)    
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))

    rank = smooth_rank_measure(all_embeds_tensor)  
    results_dict = {
        "embeds": all_embeds,
        "slide_ids": all_slide_ids
    }
    return results_dict, rank


def extract_wsi_embs_and_save(ssl_model, features_path, save_fname):

    test_dataset = SlideDataset(features_path=features_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_slide)
    results_dict, val_rank = inference_loop(ssl_model, test_dataloader)
    print("Rank = {}".format(val_rank))
    save_pkl(save_fname, results_dict)
    
    return results_dict
