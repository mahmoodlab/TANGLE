import argparse


def process_args():

    parser = argparse.ArgumentParser(description='Configurations for TANGLE pretraining')

    #----> data/splits args
    parser.add_argument('--study', type=str, default='brca', help='Study: brca or nsclc')

    #-----> model args 
    parser.add_argument('--embedding_dim', type=int, default=768, help='Size of the embedding space')
    parser.add_argument('--rna_encoder', type=str, default="mlp", help='MLP or Linear.')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--sampling_strategy', type=str, default="random", help='How to draw patch embeddings.')

    #----> training args
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--gpu_devices', type=list, default=[0,1,2], help='List of GPUs.')
    parser.add_argument('--intra_modality_mode_wsi', type=str, default='reconstruct_masked_emb', help='Type of Intra loss. Options are: reconstruct_avg_emb, reconstruct_masked_emb.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--n_tokens', type=int, default=2048, help='Number of patches to sample during training.')
    parser.add_argument('--symmetric_cl', type=bool, default=True, help='If use symmetric contrastive objective.')
    parser.add_argument('--method', type=str, default='tangle', help='Train recipe. Options are: tangle, tanglerec, intra.')
    parser.add_argument('--num_workers', type=int, default=20, help='number of cpu workers')

    args = parser.parse_args()

    return args