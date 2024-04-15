import os
import numpy as np
import time
import json

from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from core.loss.tangle_loss import InfoNCE, init_intra_wsi_loss_function
from core.models.mmssl import MMSSL
from core.dataset.dataset import TangleDataset
from core.utils.learning import collate_tangle, smooth_rank_measure
from core.utils.process_args import process_args
from core.downstream.downstream import extract_downstream_slide_embeddings

import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COHORTS = ["acc", "blca", "brca", "cesc", "chol", "coadread", "dlbc", "esca", "gbmlgg", "hnsc",
           "kidney", "lihc", "lung", "meso", "ov", "paad", "pcpg", "prad", "sarc", "skcm",
           "stad", "tgct", "thca", "thym", "ucec", "ucs", "uvm"]
LABELS = {}
for idx, cohort in enumerate(COHORTS):
    LABELS[cohort] = idx

OP108_TASK = "OncoTreeCode_label"


def train_loop(config, loss_fn, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, n_views):
        
    ssl_model.train()
    ssl_model.to(DEVICE)

    ep_loss = 0.
    fb_time = 0.
    all_embeds = []
    accum_iter = config['accum_iter']
    
    for b_idx, (patch_emb, rna_seq, avg_patch_emb) in enumerate(dataloader):
        
        losses = []
        s_fb = time.time()

        # preprocessing for intra-modality loss 
        if config["intra_modality_wsi"]:
            raise NotImplementedError('Revise implementation by return different views in dataloader.')

        # set data on device and set to float-16. 
        patch_emb = patch_emb.to(DEVICE) 
        if isinstance(rna_seq, dict):
            rna_seq = {
                'gene_ids': rna_seq['gene_ids'].to(DEVICE),  
                'gene_expression': rna_seq['gene_expression'].to(DEVICE).float(),
                'padding': rna_seq['padding'].to(DEVICE),
            }
        elif isinstance(rna_seq, torch.Tensor):
            rna_seq = rna_seq.to(DEVICE)

        if config["intra_modality_mode_wsi"] == "contrast_avg_emb" or config["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or config["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
            avg_patch_emb = avg_patch_emb.cuda()
                
        # forward pass 
        patch_emb = patch_emb.to(config['dtype'])
        rna_seq = rna_seq.to(config['dtype'])
        if config["intra_modality_wsi"]:
            out = ssl_model(patch_emb, None)
        else:
            out = ssl_model(patch_emb, rna_emb=rna_seq, n_views=n_views)
        wsi_emb, rna_emb, rna_reconstruction = out["wsi_emb"], out["rna_emb"], out["rna_reconstruction"]
        
        # inter modality loss wsi <-> rna
        if rna_emb is not None:
            if n_views > 1:
                rna_emb = rna_emb.repeat_interleave(n_views, dim=0)
                bs = int(wsi_emb.shape[0] / n_views)
                # Sub-optimal implementation bc the positive are not attracted. 
                # It's only 1 pos and bs neg repeated the number of views 
                for v in range(n_views):
                    indices = np.array(list(range(0, bs*n_views, n_views))) + v
                    losses.append(loss_fn['inter_modality'](query=wsi_emb[indices, :], positive_key=rna_emb[indices, :], symmetric=config["symmetric_cl"]))
            else:
                if config['loss'] == "info-nce":
                    curr_loss = loss_fn['inter_modality'](query=wsi_emb, positive_key=rna_emb, symmetric=config["symmetric_cl"])
                elif config['loss'] == "siglip":
                    logit_scale = out["logit_scale"]
                    logit_bias = out["logit_bias"]
                  
                    curr_loss = loss_fn['inter_modality'](image_features=wsi_emb, rna_features=rna_emb, logit_scale=logit_scale, logit_bias=logit_bias)
                elif config['loss'] == 'BarlowTwins':
                    curr_loss = loss_fn['inter_modality'](z1=wsi_emb, z2=rna_emb)
                losses.append(curr_loss)
            
        # reconstruction loss 
        if rna_reconstruction is not None:
            losses.append(loss_fn['expression_reconstruction'](rna_reconstruction, rna_seq))
            ep_recon_loss += losses[-1].item()
            
        loss = sum(losses) / n_views
        loss = loss / accum_iter
        
        # accumate loss
        loss.backward() 
        
        optimizer.step()
        optimizer.zero_grad()
                
        e_fb = time.time()
        fb_time += e_fb - s_fb

        if epoch <= config["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 5) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")
            
        ep_loss += loss.item()
        
        # get the train embeds to calculate rank
        ssl_model.eval()
        # do everything without grads 
        with torch.no_grad():
            out = ssl_model(patch_emb)
            wsi_emb_to_store = out["wsi_emb"]
            all_embeds.extend(wsi_emb_to_store.float().cpu().detach().numpy())
        ssl_model.train()

    # track rank
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    rank = smooth_rank_measure(all_embeds_tensor)  
    return ep_loss, rank


def write_dict_to_config_file(config_dict, json_file_path):
    """
    Write a dictionary to a configuration file.

    Args:
        config_dict (dict): The dictionary to be written to the config file.
        config_file_path (str): The path to the configuration file.

    Returns:
        None
    """
    config_dict_dump = {}
    for key in config_dict:
        config_dict_dump[key] = str(config_dict[key])
    
    with open(json_file_path, 'w') as jsonfile:
        json.dump(config_dict_dump, jsonfile, indent=4)


def get_args():
    args = process_args()
    args = vars(args)

    # hparams to vary 
    METHOD = 'tangle'
    STUDY = 'pancancerTCGA'
    RNA_RECONSTRUCTION = True if METHOD == 'tanglerec' else False 
    INTRA_MODALITY = True if METHOD == 'intra' else False 
    STOPPING_CRITERIA = 'train_rank' if METHOD == 'tangle' or METHOD == 'intra' else 'fixed'

    args["objective"] = METHOD
    args["rna_reconstruction"] = RNA_RECONSTRUCTION
    args["intra_modality_wsi"] = INTRA_MODALITY
    args["stopping_criteria"] = STOPPING_CRITERIA
    args["study"] = STUDY
    args["gpu_devices"] = [0, 1, 2]
    args['feature_type'] = "uni_feats"
    
    # if pancancer then add all disease models 
    if "pancancer" in STUDY:
        args["cohorts"] = COHORTS
    else:
        args["cohorts"] = [STUDY]
    
    # get dtype 
    if args['dtype'] == "float64":
        args['dtype'] = torch.float64
    elif args['dtype'] == "float32":
        args['dtype'] = torch.float32
    elif args['dtype'] == "float16":
        args['dtype'] = torch.float16
    elif args['dtype'] == "bfloat16":
        args['dtype'] = torch.bfloat16
    
    return args


def define_loss_functions(args):
    
    if args['loss'] == "info-nce":
        loss_fn_interMod = InfoNCE(temperature=args["temperature"])
    else:
        raise NotImplementedError('Only info-nce is implemented.')

    loss_fn_rnaRecon = nn.MSELoss()
    loss_fn_intraMod = init_intra_wsi_loss_function(args) 
    losses = {
        'inter_modality': loss_fn_interMod,
        'expression_reconstruction': loss_fn_rnaRecon,
        'intra_modality': loss_fn_intraMod
    }
    return losses 


if __name__ == "__main__":
    
    # setup args
    args = get_args()
    
    # paths
    ROOT_SAVE_DIR = "results/{}_checkpoints_and_embeddings".format(args["study"])
    ROOT_DATA_DIR = "data/"
    EXP_CODE = "{}lr{}_epochs{}_bs{}_tokensize{}_temperature_{}_rna{}_dtype{}_nHeads{}_accumIter{}_nViews{}_endLR{}_loss{}_hidDim{}_L2{}_rna{}_rnaNorm{}".format(
        args["name"],
        args["learning_rate"], 
        args["epochs"], 
        args["batch_size"], 
        args["token_size"],
        args["temperature"],
        args['rna_encoder'],
        args['dtype'],
        args['n_heads'],
        args['accum_iter'],
        args['n_views'],
        args['end_learning_rate'],
        args['loss'],
        args['hidden_dim'],
        args['weight_decay'],
        args['rna_data_type'],
        args['rna_normalization']
    )
    RESULS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE)
    ROOT_RNA_DIR = "rna_data/processed_data_{}/".format(args["rna_encoder"]) 
    
    # check if experiment already exists and can be skipped 
    if os.path.exists(os.path.join(RESULS_SAVE_PATH, "op108_results_dict.pkl")):
        print("{} already done, so moving on...".format(EXP_CODE))
        exit()
    
    os.makedirs(RESULS_SAVE_PATH, exist_ok=True)
    print(f"Running experiment {EXP_CODE}...")
    
    # save experiment params 
    write_dict_to_config_file(args, os.path.join(RESULS_SAVE_PATH, "config.json"))
    
    # make the datasets: Multimodal, Slide train and Slide external to derive the embeddings. 
    print("* Setup dataset...")
    
    all_datasets = []
    all_labels = []
    for cohort in args["cohorts"]:
        
        feats_dir = os.path.join(ROOT_DATA_DIR, 'tcga', cohort, args["feature_type"])
        rna_dir = os.path.join(ROOT_DATA_DIR, 'tcga', cohort, "rna_data", args['rna_data_type'])
        
        curr_dataset = TangleDataset(
            feats_dir=feats_dir,
            coords_dir=None,
            rna_dir=rna_dir,
            sampling_strategy=args["sampling_strategy"], 
            n_tokens=args["token_size"],
            normalize_rna=args['rna_normalization']
        )
        all_datasets.append(curr_dataset)
        all_labels = all_labels + [LABELS[cohort]] * len(curr_dataset)
        
    dataset = ConcatDataset(all_datasets)
    all_labels = np.array(all_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # set up dataloader
    print("* Setup dataloader...")
    if args['weighted_sample'] == "yes":
        sampler = WeightedRandomSampler(weights=class_weights_tensor, num_samples=len(dataset), replacement=True)
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], drop_last=True, collate_fn=collate_tangle, sampler=sampler)
        print(" * Using weighted sampling")
    else:
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], drop_last=True, shuffle=True, collate_fn=collate_tangle)
        print(" * Using random sampling")
     
    # set up model config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim
    print("* Setup model...")
    ssl_model = MMSSL(
        config=args,
        n_tokens_rna=args["rna_token_dim"] ,
    ).to(DEVICE).to(args['dtype']) 
    total_params = sum(p.numel() for p in ssl_model.parameters())
    print("* Total number of parameters = {}".format(total_params))
    
    if len(args["gpu_devices"]) > 1:
        print(f"* Using {torch.cuda.device_count()} GPUs.")
        ssl_model = nn.DataParallel(ssl_model, device_ids=args["gpu_devices"])
    ssl_model.to("cuda:0")
    
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = optim.AdamW(ssl_model.parameters(), lr=args["learning_rate"], weight_decay=args['weight_decay'])
    
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
    loss_fn = define_loss_functions(args)
    
    # main training loop
    best_rank = 0.
    for epoch in range(args["epochs"]):
        
        print(f"Training for epoch {epoch}...")
        
        # train
        start = time.time()
        ep_loss, train_rank = train_loop(args, loss_fn, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, args['n_views'])
        last_lr = scheduler.get_last_lr()
        end = time.time()

        print(f"Done with epoch {epoch}")
        print(f"Total loss = {ep_loss}")
        print(f"Train rank = {train_rank}")
        print(f"Last lr = {last_lr}")
        print("Total time = {:.3f} seconds".format(end-start))

        # Stop training based on rank of the training samples. Ok for TANGLE and Intra. 
        if args["stopping_criteria"] == 'train_rank' and train_rank > best_rank:
            print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
            best_rank = train_rank
            torch.save(ssl_model.state_dict(), os.path.join(RESULS_SAVE_PATH, "model.pt"))
        # Otherwise, stop after fixed number of training epochs. Ok for TANGLE-Rec. 
        else:
            torch.save(ssl_model.state_dict(), os.path.join(RESULS_SAVE_PATH, "model.pt"))
        print()

    # restore the best wsi embedder for testing.  
    print("* Loading best model...")
    ssl_model.load_state_dict(torch.load(os.path.join(RESULS_SAVE_PATH, "model.pt")))
    ssl_model.to(args['dtype'])

    # extract downstream slide embeddings using the freshly trained model
    extract_downstream_slide_embeddings(args, ssl_model, ROOT_DATA_DIR, RESULS_SAVE_PATH)
    