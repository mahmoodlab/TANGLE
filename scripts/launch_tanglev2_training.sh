
# Train Tangle on the all TCGA cohorts (referred to as Tanglev2). 
CUDA_VISIBLE_DEVICES=0,1,2 python train_tangle_pancancer.py \
    --study tanglev2 \
    --feature_type uni_feats \
    --embedding_dim 1024 \
    --rna_enc mlp \
    --wsi_encoder abmil_mh \
    --n_heads 2 \
    --hidden_dim 512 \
    --batch_size 200 \
    --n_tokens 2048 \
    --num_workers 16 \
    --weight_decay 0.01
    

    
    

