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


# def extract_downstream_slide_embeddings(args, ssl_model, root_data_dir, results_save_path):
#     print("* Computing and store MGB Lung embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/mgb_{}.csv".format('lung'),
#         features_path="{}/downstream/{}/{}".format(root_data_dir, args["feature_type"], 'lung'),
#         save_fname=os.path.join(results_save_path, "mgb_lung_results_dict.pkl"),
#     )
    
#     print("* Computing and store MGB BRCA embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/mgb_{}.csv".format('brca'),
#         features_path="{}/downstream/{}/{}".format(root_data_dir, args["feature_type"], 'brca'),
#         save_fname=os.path.join(results_save_path, "mgb_brca_result_dict.pkl"),
#     )
    
#     print("* Computing and store OP108 train embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/op108_train.csv",
#         features_path="{}/downstream/{}/{}".format(root_data_dir, args["feature_type"], 'op108_train'),
#         save_fname=os.path.join(results_save_path, "op108Train_results_dict.pkl"),
#     )
    
#     print("* Computing and store OP108 test embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/op108_test.csv",
#         features_path="{}/downstream/{}/{}".format(root_data_dir, args["feature_type"], 'op108_test'),
#         save_fname=os.path.join(results_save_path, "op108Test_results_dict.pkl"),
#     )
    
#     print("* Compute and store EBRAINS embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/{}.csv".format('ebrains'),
#         features_path="{}/downstream/ebrains/{}/".format(
#             root_data_dir, args["feature_type"],
#         ),
#         save_fname=os.path.join(results_save_path, "ebrains_results_dict.pkl"),
#         ext=".h5"
#     )
#     print()
    
#     print("* Compute and store PANDA train embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/{}_train.csv".format('panda'),
#         features_path="{}/downstream/panda/{}/".format(
#             root_data_dir, args["feature_type"],
#         ),
#         save_fname=os.path.join(results_save_path, "PandaTrain_results_dict.pkl"),
#         ext=".h5"
#     )
#     print()
    
#     print("* Compute and store PANDA test embeddings...")
#     _ = extract_wsi_embs_and_save(
#         args,
#         ssl_model=ssl_model,
#         csv_path="dataset_csv/{}_test.csv".format('panda'),
#         features_path="{}/downstream/panda/{}/".format(
#             root_data_dir, args["feature_type"],
#         ),
#         save_fname=os.path.join(results_save_path, "PandaTest_results_dict.pkl"),
#         ext=".h5"
#     )
#     print()
