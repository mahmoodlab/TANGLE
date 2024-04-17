
"""
python extract_slide_embeddings_with_pancancer_pretrained_model.py --pretrained results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/
"""

import os
import json
from collections import OrderedDict

import torch

from core.models.mmssl import MMSSL
from core.downstream.downstream import extract_wsi_embs_and_save
from core.utils.process_args import process_args

import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_args(args, config_from_model):
    exp_code = args['pretrained'].split('/')[-1]
    args['study'] = exp_code.split('_')[0]
    for key in ['wsi_encoder', 'activation', 'method', 'n_heads', 'hidden_dim', 'rna_encoder', 'rna_token_dim', 'embedding_dim', 'downstream_paths']:
        args[key] = config_from_model[key]

    args["rna_reconstruction"] = True if args["method"] == 'tanglerec' else False 
    args["intra_modality_wsi"] = True if args["method"] == 'intra' else False 
    return args 

def read_config(path_to_config):
    with open(os.path.join(path_to_config, 'config.json')) as json_file:
        data = json.load(json_file)
        return data 
 
def restore_model(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    return model 

if __name__ == "__main__":
    
    args = process_args()
    args = vars(args)
    assert args['pretrained'] is not None, "Must provide a path to a pretrained dir."
    config_from_model = read_config(args['pretrained'])
    args = set_args(args, config_from_model)

    # set up model config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim
    print("* Setup model...")
    model = MMSSL(
        config=args,
        n_tokens_rna=args["rna_token_dim"],
    ).to(DEVICE) 
    total_params = sum(p.numel() for p in model.parameters())
    print("* Total number of parameters = {}".format(total_params))
        
    # restore wsi embedder for downstream slide embedding extraction.  
    print("* Loading model from {}...".format(args['pretrained']))
    model = restore_model(model, torch.load(os.path.join(args["pretrained"], 'model.pt')))

    # extract downstream slide embeddings using the freshly trained model
    for key, val in config_from_model['downstream_paths'].items():
        print('Extracting slide embeddings in :', key)

        _ = extract_wsi_embs_and_save(
            ssl_model=model,
            features_path=val,
            save_fname=os.path.join(args["pretrained"], "{}_results_dict.pkl".format(key)),
        )
    