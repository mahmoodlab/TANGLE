import os
import numpy as np
import time
import json

from sklearn.utils.class_weight import compute_class_weight

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

import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COHORTS = ["acc", "blca", "brca", "cesc", "chol", "coadread", "dlbc", "esca", "gbmlgg", "hnsc",
           "kidney", "lihc", "lung", "meso", "ov", "paad", "pcpg", "prad", "sarc", "skcm",
           "stad", "tgct", "thca", "thym", "ucec", "ucs", "uvm"]
LABELS = {}
for idx, cohort in enumerate(COHORTS):
    LABELS[cohort] = idx


def train_loop(config, loss_fn, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):
        
    ssl_model.train()
    ssl_model.to(DEVICE)

    ep_loss = 0.
    fb_time = 0.
    all_embeds = []
    
    for b_idx, (patch_emb, rna_seq, _, _) in enumerate(dataloader):
        
        s_fb = time.time()

        # preprocessing for intra-modality loss 
        if config["intra_modality_wsi"]:
            raise NotImplementedError('Revise implementation by return different views in dataloader.')

        # set data on device and set to float-16. 
        patch_emb = patch_emb.to(DEVICE).to(config['dtype'])
        rna_seq = rna_seq.to(DEVICE).to(config['dtype'])
                
        # forward pass 
        wsi_emb, rna_emb, _ = ssl_model(patch_emb, rna_emb=rna_seq)

        # inter modality loss wsi <-> rna
        loss = loss_fn['inter_modality'](query=wsi_emb, positive_key=rna_emb, symmetric=config["symmetric_cl"])
        
        # accumate loss
        loss.backward() 
        
        optimizer.step()
        optimizer.zero_grad()
                
        e_fb = time.time()
        fb_time += e_fb - s_fb

        # step scheduler
        if epoch <= config["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 5) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")
            
        ep_loss += loss.item()
        
        # save the wsi_emb 
        all_embeds.extend(wsi_emb.float().cpu().detach().numpy())

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
    RNA_RECONSTRUCTION = True if args["study"] == 'tanglerec' else False 
    INTRA_MODALITY = True if args["study"] == 'intra' else False 
    STOPPING_CRITERIA = 'train_rank' if args["study"] == 'tangle' or args["study"] == 'intra' or args["study"] == 'tanglev2' else 'fixed'
    RNA_TOKEN_DIM = 5248

    args["objective"] = args["study"]
    args["rna_reconstruction"] = RNA_RECONSTRUCTION
    args["intra_modality_wsi"] = INTRA_MODALITY
    args["stopping_criteria"] = STOPPING_CRITERIA
    args["study"] = args["study"]
    args["gpu_devices"] = [int(x) for x in range(torch.cuda.device_count())]
    args["cohorts"] = COHORTS
    args["rna_token_dim"] = RNA_TOKEN_DIM

    # set loss -- here, we only provide info-nce
    args['loss'] = "info-nce"
    
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
    ROOT_SAVE_DIR = "./results/{}_checkpoints_and_embeddings".format(args["study"])
    ROOT_DATA_DIR = "./data/tcga"

    EXP_CODE = "{}lr{}_epochs{}_bs{}_tokensize{}_temperature_{}_rna{}_dtype{}_nHeads{}_endLR{}_loss{}_hidDim{}_L2{}".format(
        args["study"],
        args["learning_rate"], 
        args["epochs"], 
        args["batch_size"], 
        args["n_tokens"],
        args["temperature"],
        args['rna_encoder'],
        args['dtype'],
        args['n_heads'],
        args['end_learning_rate'],
        args['loss'],
        args['hidden_dim'],
        args['weight_decay'],
    )
    RESULS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE)
    
    os.makedirs(RESULS_SAVE_PATH, exist_ok=True)
    print(f"Running experiment {EXP_CODE}...")
    
    # save experiment params 
    write_dict_to_config_file(args, os.path.join(RESULS_SAVE_PATH, "config.json"))
    
    # make the datasets: Multimodal, Slide train and Slide external to derive the embeddings. 
    print("* Setup dataset...")
    
    all_datasets = []
    all_labels = []
    for cohort in args["cohorts"]:
        feats_dir = os.path.join(ROOT_DATA_DIR, cohort, args["feature_type"])
        rna_dir = os.path.join(ROOT_DATA_DIR, cohort, "molecular_data", "normed")
        
        curr_dataset = TangleDataset(
            feats_dir=feats_dir,
            rna_dir=rna_dir,
            sampling_strategy=args["sampling_strategy"], 
            n_tokens=args["rna_token_dim"],
        )
        all_datasets.append(curr_dataset)
        
    dataset = ConcatDataset(all_datasets)
    print("* Training dataset size = {}".format(len(dataset)))
    
    # set up dataloader
    print("* Setup dataloader...")
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], drop_last=True, shuffle=True, collate_fn=collate_tangle)
    
    # set up model config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim
    print("* Setup model...")
    ssl_model = MMSSL(
        config=args,
        n_tokens_rna=args["rna_token_dim"],
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
        ep_loss, train_rank = train_loop(args, loss_fn, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler)
        last_lr = scheduler.get_last_lr()
        end = time.time()

        print(f"Done with epoch {epoch}")
        print(f"Total loss = {ep_loss}")
        print(f"Train rank = {train_rank}")
        print(f"Last lr = {last_lr}")
        print("Total time = {:.3f} seconds".format(end-start))

        # Stop training based on rank of the training samples. Ok for TANGLE and Intra. 
        if args["stopping_criteria"] == 'train_rank':
            if train_rank > best_rank:
                print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
                best_rank = train_rank
                torch.save(ssl_model.state_dict(), os.path.join(RESULS_SAVE_PATH, "model.pt"))
        # Otherwise, stop after fixed number of training epochs. Ok for TANGLE-Rec. 
        else:
            torch.save(ssl_model.state_dict(), os.path.join(RESULS_SAVE_PATH, "model.pt"))
        print()
