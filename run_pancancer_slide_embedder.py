
import os
import json

import torch
import torch.nn as nn

from core.models.mmssl import MMSSL
from core.downstream.downstream import extract_downstream_slide_embeddings

import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    # setup args
    args = read_config()
    
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

    MODEL_PATH = 'results/pancancer_checkpoints_and_embeddings/'
    ROOT_DATA_DIR = "data/"

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
    
    # restore wsi embedder for downstream slide embedding extraction.  
    print("* Loading best model...")
    ssl_model.load_state_dict(torch.load(os.path.join(MODEL_PATH)))
    ssl_model.to(args['dtype'])

    # extract downstream slide embeddings using the freshly trained model
    extract_downstream_slide_embeddings(args, ssl_model, ROOT_DATA_DIR, RESULS_SAVE_PATH)
