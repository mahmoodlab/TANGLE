import argparse


def process_args():

    parser = argparse.ArgumentParser(description='Configurations for TANGLE pretraining')

    #----> Tangle (BRCA or NSCLC) vs Tanglev2 (Pancancer)
    parser.add_argument('--study', type=str, default='brca', help='Study: brca, nsclc or pancancer')

    #-----> model args 
    parser.add_argument('--embedding_dim', type=int, default=768, help='Size of the embedding space')
    parser.add_argument('--rna_encoder', type=str, default="mlp", help='MLP or Linear.')
    parser.add_argument('--sampling_strategy', type=str, default="random", help='How to draw patch embeddings.')
    parser.add_argument('--wsi_encoder', type=str, default="abmil", help='Type of MIL.')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of heads in ABMIL.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Internal dim of ABMIL.')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function used in ABMIL attention weight agg (sigmoid or softmax).')
    parser.add_argument('--mask_percentage', type=float, default=0.5, help='Percentage of n_tokens that is masked during Intra loss computation.')

    #----> training args
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Tensor dtype. Defaults to bfloat16 for increased batch size.')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--gpu_devices', type=list, default=[0], help='List of GPUs.')
    parser.add_argument('--intra_modality_mode_wsi', type=str, default='reconstruct_masked_emb', help='Type of Intra loss. Options are: reconstruct_avg_emb, reconstruct_masked_emb.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--n_tokens', type=int, default=2048, help='Number of patches to sample during training.')
    parser.add_argument('--symmetric_cl', type=bool, default=True, help='If use symmetric contrastive objective.')
    parser.add_argument('--method', type=str, default='tangle', help='Train recipe. Options are: tangle, tanglerec, intra.')
    parser.add_argument('--num_workers', type=int, default=20, help='number of cpu workers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')

    #---> model inference 
    parser.add_argument('--pretrained', type=str, default=None, help='Path to dir with checkpoint.')

    args = parser.parse_args()

    return args