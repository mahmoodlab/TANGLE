# --> General imports
import os
import numpy as np
import pandas as pd
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

# --> internal imports 
from core.models.mmssl import MMSSL
from core.dataset.dataset import TangleDataset, SlideDataset
from core.loss.tangle_loss import InfoNCE, apply_random_mask, init_intra_wsi_loss_function
from core.utils.learning import smooth_rank_measure, collate_tangle, collate_slide, set_seed
from core.utils.process_args import process_args

import pdb

# Set device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(args, loss_fn_interMod, loss_fn_rnaRecon, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):
        
    ssl_model.train()
    ssl_model.to(DEVICE)

    ep_loss, ep_recon_loss, ep_inter_loss, ep_intra_loss = 0., 0., 0., 0.
    fb_time = 0.
    all_embeds = []
    
    for b_idx, (patch_emb, rna_seq, patch_emb_aug, avg_patch_emb) in enumerate(dataloader):
        
        losses = []    
        s_fb = time.time()

        # preprocessing for intra-modality loss 
        if args["intra_modality_wsi"]:
            if args["intra_modality_mode_wsi"] == "contrast_token_views":
                patch_emb = torch.cat((patch_emb, patch_emb_aug))
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb" or args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
                patch_emb_mask = apply_random_mask(patch_embeddings=patch_emb, percentage=args['mask_percentage'])
                patch_emb = torch.cat((patch_emb, patch_emb_mask))

        # set data on device 
        patch_emb = patch_emb.to(DEVICE)
        rna_seq = rna_seq.to(DEVICE) if rna_seq is not None else rna_seq
        if args["intra_modality_mode_wsi"] == "contrast_avg_emb" or args["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
            avg_patch_emb = avg_patch_emb.cuda()
                
        # forward pass and loss 
        if args["intra_modality_wsi"]:
            wsi_emb, rna_emb, rna_reconstruction = ssl_model(patch_emb, None)
        else:
            wsi_emb, rna_emb, rna_reconstruction = ssl_model(patch_emb, rna_seq)
        
        # intra modality loss wsi <-> wsi
        if rna_emb is None and rna_reconstruction is None:
            if args["intra_modality_mode_wsi"] == "contrast_token_views":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(query=wsi_emb[:split_idx], positive_key=wsi_emb[split_idx:], symmetric=args["symmetric_cl"])) # 1. first set of token views 2. second set of token views (augmentation)
            elif args["intra_modality_mode_wsi"] == "contrast_avg_emb":
                losses.append(loss_fn_intraMod(query=wsi_emb, positive_key=avg_patch_emb, symmetric=args["symmetric_cl"]))
            elif args["intra_modality_mode_wsi"] == "reconstruct_avg_emb":
                losses.append(loss_fn_intraMod(wsi_emb, avg_patch_emb))
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(wsi_emb[split_idx:], wsi_emb[:split_idx])) # 1. masked wsi_emb 2. umasked wsi_emb
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(wsi_emb[split_idx:], wsi_emb[:split_idx])) # 1. masked wsi_emb 2. umasked wsi_emb
                losses.append(loss_fn_intraMod(query=wsi_emb[:split_idx], positive_key=avg_patch_emb, symmetric=args["symmetric_cl"]))
            else:
                raise ValueError("Invalid intra_modality_mode_wsi.")
            ep_intra_loss += losses[-1].item()
            
        # inter modality loss wsi <-> rna
        if rna_emb is not None:
            losses.append(loss_fn_interMod(query=wsi_emb, positive_key=rna_emb, symmetric=args["symmetric_cl"]))
            ep_inter_loss += losses[-1].item()
            
        # intra modality loss rna <-> rna
        if rna_reconstruction is not None:
            losses.append(loss_fn_rnaRecon(rna_reconstruction, rna_seq))
            ep_recon_loss += losses[-1].item()
            
        loss = sum(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        e_fb = time.time()
        fb_time += e_fb - s_fb

        if epoch <= args["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")
            
        ep_loss += loss.item()
        
        # get the train embeds to calculate rank
        ssl_model.eval()
        # do everything without grads 
        with torch.no_grad():
            wsi_emb_to_store, _, _ = ssl_model(patch_emb)
            all_embeds.extend(wsi_emb_to_store.detach().cpu().numpy())
        ssl_model.train()
    
    # track rank
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    rank = smooth_rank_measure(all_embeds_tensor)  
        
    return ep_loss, rank


def val_loop(ssl_model, val_dataloader):
    
    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)
    
    all_embeds = []
    all_labels = []
    
    # do everything without grads 
    with torch.no_grad():
        
        for inputs, labels in tqdm(val_dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            wsi_embed, _, _ = ssl_model(inputs)
            wsi_embed = wsi_embed.detach().cpu().numpy()
            all_embeds.extend(wsi_embed)
            all_labels.append(labels.item())
            
    all_embeds = np.array(all_embeds)
    all_labels = np.array(all_labels)
    
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    rank = smooth_rank_measure(all_embeds_tensor)  
    results_dict = {"embeds": all_embeds, "labels": all_labels}
    
    return results_dict, rank


if __name__ == "__main__":
    
    # setup args and seed
    args = process_args()
    args = vars(args)
    set_seed(args["seed"])
    
    RNA_RECONSTRUCTION = True if args["method"] == 'tanglerec' else False 
    INTRA_MODALITY = True if args["method"] == 'intra' else False 
    STOPPING_CRITERIA = 'train_rank' if args["method"] == 'tangle' or args["method"] == 'intra' else 'fixed'
    N_TOKENS_RNA = 4908 if args["study"]=='nsclc' else 4999

    args["rna_reconstruction"] = RNA_RECONSTRUCTION
    args["intra_modality_wsi"] = INTRA_MODALITY

    # paths 
    ROOT_SAVE_DIR = "./results/{}_checkpoints_and_embeddings".format(args["study"])
    EXP_CODE = "{}_{}_lr{}_epochs{}_bs{}_tokensize{}_temperature{}".format(
        args["method"],
        args["study"],
        args["learning_rate"], 
        args["epochs"], 
        args["batch_size"], 
        args["n_tokens"],
        args["temperature"]
    )
    RESULTS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE)

    print()
    print(f"Running experiment {EXP_CODE}...")
    print()
    
    # Create a SummaryWriter
    log_dir = os.path.join(ROOT_SAVE_DIR, 'logs', EXP_CODE)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    
    # make tangle dataset
    print("* Setup dataset...")
    dataset = TangleDataset(
        feats_dir="./data/{}/ctranspath_features/tcga_features/".format(args["study"]), 
        rna_dir='./data/{}/rna'.format(args["study"]), 
        sampling_strategy=args["sampling_strategy"], 
        n_tokens=args["n_tokens"]
    )
    
    # in-domain dataset loader for extracting in-domain slide embeddings. 
    PATH_TO_INDOMAIN_FEATS = "./data/{}/ctranspath_features/tcga_features/".format(args["study"])
    PATH_TO_INDOMAIN_CSV = "./data/{}/csvs/tcga_{}.csv".format(args["study"], 'lung' if args["study"]=='nsclc' else 'brca')
    indomain_dataset = SlideDataset(
        csv_path=PATH_TO_INDOMAIN_CSV, 
        features_path=PATH_TO_INDOMAIN_FEATS
    )

    # out-of-domain dataset loader for extracting out-of-domain slide embeddings. 
    PATH_TO_OUOFDOMAIN_FEATS = "./data/{}/ctranspath_features/mgb_features/".format(args["study"])
    PATH_TO_OUTOFDOMAIN_CSV = "./data/{}/csvs/op_{}.csv".format(args["study"], 'lung' if args["study"]=='nsclc' else 'brca')
    outofdomain_dataset = SlideDataset(
        csv_path=PATH_TO_OUTOFDOMAIN_CSV, 
        features_path=PATH_TO_OUOFDOMAIN_FEATS
    )
    
    # set up all dataloaders
    print("* Setup dataloader...")
    dataloader = DataLoader(
        dataset, 
        batch_size=args["batch_size"], 
        shuffle=True, 
        collate_fn=collate_tangle
    )
    indomain_dataloader = DataLoader(indomain_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_slide)
    outofdomain_dataloader = DataLoader(outofdomain_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_slide)
    
    # set up model config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim=768
    print("* Setup model...")
    
    ssl_model = MMSSL(config=args, n_tokens_rna=N_TOKENS_RNA).to(DEVICE)
    
    if len(args["gpu_devices"]) > 1:
        print(f"* Using {torch.cuda.device_count()} GPUs.")
        ssl_model = nn.DataParallel(ssl_model, device_ids=args["gpu_devices"])
    ssl_model.to("cuda:0")
    
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = optim.AdamW(ssl_model.parameters(), lr=args["learning_rate"])
    
    # set up schedulers
    print("* Setup schedulers...")
    T_max = (args["epochs"] - args["warmup_epochs"]) * len(dataloader) if args["warmup"] else args["epochs"] * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=T_max,
        eta_min=args["end_learning_rate"]
    )
    
    if args["warmup"]:
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.00001,
            total_iters=args["warmup_epochs"] * len(dataloader)
    )
    else:
        scheduler_warmup = None
    
    # set up losses
    print("* Setup losses...")
    loss_fn_interMod = InfoNCE(temperature=args["temperature"])
    loss_fn_rnaRecon = nn.MSELoss()
    loss_fn_intraMod = init_intra_wsi_loss_function(args) 

    # main training loop
    best_rank = 0.
    for epoch in range(args["epochs"]):
        
        print()
        print(f"Training for epoch {epoch}...")
        print()
        
        # train
        start = time.time()
        ep_loss, train_rank = train_loop(args, loss_fn_interMod, loss_fn_rnaRecon, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler)
        writer.add_scalar('Training Loss', ep_loss, epoch)
        writer.add_scalar('Train rank', train_rank, epoch)
        end = time.time()

        print()
        print(f"Done with epoch {epoch}")
        print(f"Total loss = {ep_loss}")
        print(f"Train rank = {train_rank}")
        print("Total time = {:.3f} seconds".format(end-start))

        # Stop training based on rank of the training samples. Ok for TANGLE and Intra. 
        if STOPPING_CRITERIA == 'train_rank' and train_rank > best_rank:
            print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
            best_rank = train_rank
            torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model.pt"))
        # Otherwise, stop after fixed number of training epochs. Ok for TANGLE-Rec. 
        else:
            torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model.pt"))
        print()
    
    # save the wsi_embedder model 
    ssl_model.load_state_dict(torch.load(os.path.join(RESULTS_SAVE_PATH, "model.pt")))
    
    # get in-domain (tcga here) slide embeddings 
    print()
    print("Compute and store TCGA embeddings...")
    print()
    tcga_results_dict, val_rank = val_loop(ssl_model, val_dataloader)
    writer.add_scalar("TCGA_val_rank", val_rank)
    print("Rank = {}".format(val_rank))
        
    # get op_brca embeds 
    print()
    print("Compute and store of-of-domain embeddings...")
    print()
    outofdomain_results_dict, _ = val_loop(ssl_model, test_dataloader)
    
    # save 
    with open(os.path.join(RESULTS_SAVE_PATH, "tcga_results_dict.pkl"), 'wb') as handle:
        pickle.dump(tcga_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(RESULTS_SAVE_PATH, "outofdomain_results_dict.pkl"), 'wb') as handle:
        pickle.dump(outofdomain_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("Done")
    print()
