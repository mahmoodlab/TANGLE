from core.datasets.dataset_mmssl import BRCASlideEmbeddingDataset
from core.utils.utils_mmssl import set_determenistic_mode
from core.models.abmil import BatchedABMIL
from core.models.transmil import TransMIL

from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, auc
from tqdm import tqdm
from datetime import datetime
import os 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import pdb 
        
##################### INPUT PARAMS #####################

feature_dirs_val = [
    # LUNG TANGLE:
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # LUNG TANGLE-Rec:
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # LUNG Intra:
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # BREAST TANGLE:
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # BREAST TANGLE-REC:
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # BREAST INTRA:
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
]

feature_dirs_test = [
    # LUNG TANGLE:
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # LUNG TANGLE-Rec:
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # LUNG Intra:
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    "./results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # BREAST TANGLE:
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # BREAST TANGLE-REC:
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # BREAST TANGLE-REC:
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    "./results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
]

save_dir = './results/'
aggregation_mode = ['Average'] # Other option is 'Average', 'MIL'
few_shot_k = [1,5,10,25] # select k abnormal samples per class and k normal samples per class for training, if -1 all samples are used
seeds = [0,6,12,18,24,48,96,128,256,512] 
metrics = ["auc", "f1", "bacc"]
max_iter=10000
########################################################

current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M-%S")
results_df = pd.DataFrame()

def run_linear_probing(few_shot_datasets_train, few_shot_datasets_test, results_df, run_df_idx, model_name):
    scores = []
    # train linear binary classifier          
    x_train = few_shot_datasets_train["features"]
    y_train = few_shot_datasets_train["binary_classes"].squeeze()
    x_test = few_shot_datasets_test["features"]
    y_test = few_shot_datasets_test["binary_classes"].squeeze()
    
    clf = LogisticRegression(max_iter=max_iter).fit(x_train, y_train)
    
    if "bacc" in metrics:
        y_pred_test_prob = clf.predict_proba(x_test)[:,1]
        y_pred_test = clf.predict(x_test)
        results_df.loc[run_df_idx, "model_name"] = model_name
        results_df.loc[run_df_idx, "run"] = run_idx
        results_df.loc[run_df_idx, "k"] = f"{k}"
        results_df.loc[run_df_idx, "metric"] = "BACC"
        results_df.loc[run_df_idx, "score"] = balanced_accuracy_score(y_pred=y_pred_test, y_true=y_test)
        run_df_idx += 1
    
    if "auc" in metrics:
        y_pred_test_prob = clf.predict_proba(x_test)[:,1]
        fpr, tpr, tresholds = roc_curve(y_score=y_pred_test_prob, y_true=y_test)
        results_df.loc[run_df_idx, "model_name"] = model_name
        results_df.loc[run_df_idx, "run"] = run_idx
        results_df.loc[run_df_idx, "k"] = f"{k}"
        results_df.loc[run_df_idx, "metric"] = "AUC"
        results_df.loc[run_df_idx, "score"] = auc(fpr, tpr)
        run_df_idx += 1

    if "f1" in metrics:
        y_pred_test_prob = clf.predict_proba(x_test)[:,1]
        y_pred_test = clf.predict(x_test)
        results_df.loc[run_df_idx, "model_name"] = model_name
        results_df.loc[run_df_idx, "run"] = run_idx
        results_df.loc[run_df_idx, "k"] = f"{k}"
        results_df.loc[run_df_idx, "metric"] = "F1"
        results_df.loc[run_df_idx, "score"] = f1_score(y_pred=y_pred_test, y_true=y_test)
        run_df_idx += 1

    return results_df, run_df_idx


def print_run_metrics(results_df, model_name):
    # print results 
    print('\n\n', feature_dir_val)
    for k in few_shot_k:
        # scores = results_df[(results_df['k'] == str(k)) & (results_df['metric'] == 'BACC')]['score'].values # @TODO: update formula
        # print('BACC:', k, round(100*scores.mean(), 1), round(100*scores.std(), 1))
        scores = results_df[
            (results_df['k'] == str(k)) &
            (results_df['metric'] == 'AUC') & 
            (results_df['model_name'] == model_name)
        ]['score'].values
        print('k:', k, 'AUC:', round(100*scores.mean(), 1), '$\pm$', round(100*scores.std(), 1))
        # scores = results_df[(results_df['k'] == str(k)) & (results_df['metric'] == 'F1')]['score'].values # @TODO: update formula
        # print('F1:', k, round(100*scores.mean(), 1), round(100*scores.std(), 1))


run_df_idx = 0
# iterate over different embedding spaces / models
for feature_dir_val, feature_dir_curated_test in zip(feature_dirs_val, feature_dirs_test):
    for agg_fn in aggregation_mode:
        model_name = feature_dir_val.split("/")[-2]
        agg_fn = None
                                                
        # iterate over different k in few shot setting
        for k_idx, k in enumerate(tqdm(few_shot_k, desc=f"Experiment over different k")): 
            # different runs with different data splits
            for run_idx, seed in enumerate(seeds):

                feature_dataset_train = BRCASlideEmbeddingDataset(feature_folder_path=feature_dir_val, agg_fn=agg_fn)
                feature_dataset_test = BRCASlideEmbeddingDataset(feature_folder_path=feature_dir_curated_test, agg_fn=agg_fn)

                set_determenistic_mode(SEED=seed)
                few_shot_datasets_train = feature_dataset_train.get_few_shot_binary_datasets(k=k)
                few_shot_datasets_test = feature_dataset_test.get_few_shot_binary_datasets(k=None)

                results_df, run_df_idx = run_linear_probing(few_shot_datasets_train, few_shot_datasets_test, results_df, run_df_idx, model_name)

        print_run_metrics(results_df, model_name)
        results_df.to_csv(save_dir + "/" + formatted_time + "_" + f'{feature_dirs_test[0].split("/")[-1]}_' + str(metrics) + ".csv")

results_df.to_csv(save_dir + "/" + formatted_time + "_" + f'{feature_dirs_test[0].split("/")[-1]}_' + str(metrics) + ".csv")
