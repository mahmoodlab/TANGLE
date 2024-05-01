import pandas as pd
import torch
import os
import numpy as np
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

from core.utils.learning import set_seed

import pdb

BCNB_BREAST_TASKS = ['er', 'pr', 'her2']
BREAST_TASKS = {'BCNB': BCNB_BREAST_TASKS}


def calculate_metrics(y_true, y_pred, pred_scores):
    """
    Calculate and print various evaluation metrics.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - y_scores: Target scores (for AUC).
    """
    if len(np.unique(y_true)) > 2:
        # multi-class 
        auc = roc_auc_score(y_true, pred_scores, multi_class="ovr", average="macro",)
    else:
        # regular 
        auc = roc_auc_score(y_true, pred_scores[:, 1]) # only send positive class score)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return auc, bacc


def load_and_split(labels, embedding_path, study, k=1):

    # 1. load embeddings as dict where key is slide ID 
    file = open(embedding_path, 'rb')
    obj = pickle.load(file)
    embeddings = obj['embeds']
    slide_ids = obj['slide_ids']
    slide_ids = [str(x) for x in slide_ids]
    embeddings = {n: e for e, n in zip(embeddings, slide_ids)}

    # 2. make sure the intersection is solid. 
    intersection = list(set(labels['slide_id'].values.tolist()) & set(slide_ids))
    labels = labels[labels['slide_id'].isin(intersection)]
    num_classes = len(labels[study].unique())
    
    # 3. define random split and extract corresponding slide IDs, embeddings and labels 
    train_slide_ids = []
    for cls in range(num_classes):
        train_slide_ids += labels[labels[study] == cls].sample(k)['slide_id'].values.tolist()
    test_slide_ids = labels[~labels['slide_id'].isin(train_slide_ids)]['slide_id'].values.tolist()

    train_embeddings = np.array([embeddings[n] for n in train_slide_ids])
    test_embeddings = np.array([embeddings[n] for n in test_slide_ids])

    train_labels = np.array([labels[labels['slide_id']==slide_id][study].values for slide_id in train_slide_ids]) 
    test_labels = np.array([labels[labels['slide_id']==slide_id][study].values for slide_id in test_slide_ids])  

    # 4. make sure everything has the right format and dimensions 
    train_embeddings = torch.from_numpy(train_embeddings)
    test_embeddings = torch.from_numpy(test_embeddings)

    train_labels = torch.from_numpy(train_labels).squeeze()
    test_labels = torch.from_numpy(test_labels).squeeze()

    if len(train_embeddings.shape) == 1:
        train_embeddings = torch.unsqueeze(train_embeddings, 0)
        train_labels = torch.unsqueeze(train_labels, 0)

    return train_embeddings, train_labels, test_embeddings, test_labels
    

def eval_single_task(DATASET_NAME, TASKS, PATH, verbose=True):
        
    ALL_K = [10]
  
    if DATASET_NAME == "BCNB":
        EMBEDS_PATH = "{}/bcnb_results_dict.pkl".format(PATH) 
        LABEL_PATH = 'dataset_csv/bcnb_brca.csv'
    else:
        raise NotImplementedError("Dataset not implemented")

    BASE_OUT = '/'.join(EMBEDS_PATH.split('/')[:-1])
    
    for k in ALL_K:
        for task in TASKS:
            if verbose:
                print(f"Task {task} and k = {k}...")
            NUM_FOLDS = 10 
            metrics_store_all = {}
            RESULTS_FOLDER = f"k={k}_probing_{task.replace('/', '')}"

            metrics_store = {"auc": [], "bacc": []}
        
            # go over folds
            for fold in range(NUM_FOLDS):
                set_seed(SEED=fold)
                if verbose:
                    print(f"     Going for fold {fold}...")

                # Load and process labels  
                LABELS = pd.read_csv(LABEL_PATH) 
                LABELS['slide_id'] = LABELS['slide_id'].astype(str)
                LABELS = LABELS[LABELS[task] != -1]
                LABELS = LABELS[['slide_id', task]]

                # Load embeddings, labels and split data 
                train_features, train_labels, test_features, test_labels = load_and_split(LABELS, EMBEDS_PATH, task, k)
    
                if verbose:
                    print(f"     Fitting logistic regression on {len(train_features)} slides")
                    print(f"     Evaluating on {len(test_features)} slides")

                NUM_C = 2
                COST = (train_features.shape[1] * NUM_C) / 100
                clf = LogisticRegression(C=COST, max_iter=10000, verbose=0, random_state=0)
                # clf = LogisticRegression(max_iter=100000)
                clf.fit(X=train_features, y=train_labels)
                pred_labels = clf.predict(X=test_features)
                pred_scores = clf.predict_proba(X=test_features)

                # print metrics
                if verbose:
                    print("     Updating metrics store...")
                
                # task specific metrics 
                if task == "isup_grade":
                    weighted_kappa = cohen_kappa_score(test_labels.numpy(), pred_labels, weights='quadratic')
                    bacc = balanced_accuracy_score(test_labels.numpy(), pred_labels)
                    metrics_store["q_kappa"].append(weighted_kappa)
                    metrics_store["bacc"].append(bacc)
                else:
                    auc, bacc = calculate_metrics(test_labels.numpy(), pred_labels, pred_scores)
                    metrics_store["auc"].append(auc)
                    metrics_store["bacc"].append(bacc)
                
                if verbose:
                    print(f"     Done for fold {fold} -- AUC: {round(auc, 3)}, BACC: {round(bacc, 3)}\n")
            
            metrics_store_all['tangle'] = metrics_store
            if task == "isup_grade":
                print('k={}, task={}, quadratic kappa={}'.format(
                    k,
                    task,
                    round(np.array(metrics_store['q_kappa']).mean(), 3))
                )
            else:
                print('k={}, task={}, auc={} +/- {}'.format(
                    k,
                    task,
                    round(np.array(metrics_store['auc']).mean(), 3),
                    round(np.array(metrics_store['auc']).std(), 3)
                    )
                )
            
            # save results for plotting
            os.makedirs(f'{BASE_OUT}/{DATASET_NAME}', exist_ok=True)
            with open(f'{BASE_OUT}/{DATASET_NAME}/{RESULTS_FOLDER}.pickle', 'wb') as handle:
                pickle.dump(metrics_store_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

# main 
if __name__ == "__main__":

    tasks = BREAST_TASKS
    print("* Evaluating on breast...")
    print("* All datasets to evaluate on = {}".format(list(tasks.keys())))

    # Put your slide embeddings here... 
    MODELS = {
        'tangle_brca': "results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.1/",
        'tanglerec_brca': "results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs128_tokensize2048_temperature0.01",
        'intra_brca': "results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs128_tokensize4096_temperature0.01/",
        'tangle_pancancer': "results/pancancer_checkpoints_and_embeddings/tangle_pancancer"
    }

    for exp_name, p in MODELS.items():
        for n, t in tasks.items():
            print('\n* Dataset:', exp_name)
            eval_single_task(n, t, p, verbose=False)
