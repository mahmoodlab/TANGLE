
# --> Generic imports 
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, auc
from tqdm import tqdm
from datetime import datetime
import os 

# --> Internal imports 
from core.dataset.dataset import FewShotClassificationDataset
from core.utils.learning import set_seed


import pdb 
        
##################### INPUT PARAMS #####################

feature_dirs_train = [
    # # LUNG TANGLE:
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # # LUNG TANGLE-Rec:
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # # LUNG Intra:
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
    # BREAST TANGLE:
    "./legacy_results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/tcga_results_dict.pkl",
    # BREAST TANGLE-REC:
    "./legacy_results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/tcga_results_dict.pkl",
    # BREAST INTRA:
    "./legacy_results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/tcga_results_dict.pkl",
]

feature_dirs_test = [
    # # LUNG TANGLE:
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tangle_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # # LUNG TANGLE-Rec:
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/tanglerec_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # # LUNG Intra:
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize1024_temperature0.01/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    # "./legacy_results/nsclc_checkpoints_and_embeddings/intra_nsclc_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
    # BREAST TANGLE:
    "./legacy_results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/mgb_results_dict.pkl",
    # BREAST TANGLE-REC:
    "./legacy_results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01/mgb_results_dict.pkl",
    # BREAST INTRA:
    "./legacy_results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/mgb_results_dict.pkl",
]

save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)
few_shot_k = [1,5,10,25] # select k abnormal samples per class and k normal samples per class for training, if -1 all samples are used
seeds = [0,6,12,18,24,48,96,128,256,512] 
metrics = ["auc", "f1", "bacc"]
max_iter=10000
########################################################


def run_linear_probing(few_shot_datasets_train, few_shot_datasets_test, results_df, run_df_idx, model_name):

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
    print('\n\n', feature_dir_train)
    for k in few_shot_k:

        scores = results_df[
            (results_df['k'] == str(k)) &
            (results_df['metric'] == 'AUC') & 
            (results_df['model_name'] == model_name)
        ]['score'].values
        print('k:', k, 'AUC:', round(100*scores.mean(), 1), '$\pm$', round(100*scores.std(), 1))


if __name__ == "__main__":

    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d-%H-%M-%S")
    results_df = pd.DataFrame()

    run_df_idx = 0
    # iterate over different embedding spaces / models
    for feature_dir_train, feature_dir_test in zip(feature_dirs_train, feature_dirs_test):
        model_name = feature_dir_train.split("/")[-2]
                                                
        # iterate over different k in few shot setting
        for k_idx, k in enumerate(tqdm(few_shot_k, desc=f"Experiment over different k")): 
            # different runs with different data splits
            for run_idx, seed in enumerate(seeds):

                feature_dataset_train = FewShotClassificationDataset(feature_folder_path=feature_dir_train)
                feature_dataset_test = FewShotClassificationDataset(feature_folder_path=feature_dir_test)

                set_seed(SEED=seed)
                few_shot_datasets_train = feature_dataset_train.get_few_shot_binary_datasets(k=k)
                few_shot_datasets_test = feature_dataset_test.get_few_shot_binary_datasets(k=None)

                results_df, run_df_idx = run_linear_probing(few_shot_datasets_train, few_shot_datasets_test, results_df, run_df_idx, model_name)

        print_run_metrics(results_df, model_name)
        results_df.to_csv(save_dir + "/" + formatted_time + "_" + f'{feature_dirs_test[0].split("/")[-1]}_' + str(metrics) + ".csv")

    results_df.to_csv(save_dir + "/" + formatted_time + "_" + f'{feature_dirs_test[0].split("/")[-1]}_' + str(metrics) + ".csv")
