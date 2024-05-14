# Code for Transcriptomics-guided Slide Representation Learning in Computational Pathology

Welcome to the official GitHub repository of our CVPR 2024 paper, "Transcriptomics-guided Slide Representation Learning in Computational Pathology". This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. 

![Alt text for the image](support/framework.png "Optional title")

## Highlights
This work introduces the first method for **slide representation learning** using **multimodal pretraining**. Slide representation learning defines a new class of self-supervised methods that aim at extracting information-rich embeddings of histology whole-slide images without using explicit supervision (such as cancer subtype or cancer grade). In this work, we introduce **TANGLE**, a method for Slide + Expression (S+E) pretraining. Conceptually, this method follows the CLIP principle widely employed in Vision-Language models. Here, **we align the slide with its corresponding gene expression profile**. The resulting slide encoder embeds the underlying molecular landscape of the tissue and can, as such, be used for various downstream tasks. In this work, we focus on morphological subtyping of breast and lung cancer and morphological lesion detection in pre-clinical drug safety studies.  

## Code
This repository contains the implementation of **TANGLE**, with step-by-step instructions to pretrain a TANGLE model on the **TCGA-BRCA** (Invasive breast cancer) cohort. The resulting models is evaluated using linear probing (logistic regression) on BCNC for ER/PR/HER2 status prediction and BRACS fine and coarse subtyping. 

In addition, we provide a script for pan-cancer TANGLE pretraining that aggregates all TCGA cohorts into a unified training. This model was not part of TANGLE and is referred to as TANGLEv2. 

### Installation

```bash
# Clone repo
git clone https://github.com/mahmoodlab/TANGLE
cd TANGLE

# Create conda env
conda create -n tangle
conda activate tangle
pip install -r requirements.txt
```

### Preprocessing 

This code assumes that you have already preprocessed your dataset.

Preprocessing whole-slide images requires (1) segmenting the tissue, (2) extracting patches, and (3) extracting patch embeddings. In TCGA-BRCA, we train TANGLE using 2 patch encoder: a ResNet50 features (pretrained on ImageNet) and CTransPath. 

- Tissue segmentation, patching, and ResNet50 patch embedding extraction can be done using the [CLAM toolbox](https://github.com/mahmoodlab/CLAM).
- CTransPath patch embedding extraction can be done using their [official implementation](https://github.com/Xiyue-Wang/TransPath). 

Preprocessing the corresponding gene expression profile can be done in several ways. For the TCGA cohorts, we used normalized RNA sequencing data available [here](https://xenabrowser.net/datapages/?dataset=TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443). A more processed form can be accessed [here](https://github.com/mahmoodlab/SurvPath/blob/main/datasets_csv/raw_rna_data/combine/brca/rna_clean.csv).

### Training on TCGA-BRCA

To simplify reproducing results, we provide a link to a [Drive](https://drive.google.com/drive/folders/1GIJEITf5-7lFKil7Dfi3sSmVFgzh-otv?usp=sharing) that includes (1) TCGA-BRCA CTransPath patch embeddings, (2) the corresponding expression profiles as `pt` files. These need to be downloaded and moved to the base directory in a new dir called `data`. Due to MGB cohort being private, we cannot upload the downstream in-house MGB cohort, and instead provide results based on BCNB and BRACS.

To train Tangle, use:

```bash
# Train Tangle
source scripts/launch_tangle_training.sh
```

### Evaluate TANGLE trained with TCGA-BRCA on BRACS and BCNB

We provide a link to a [Drive](https://drive.google.com/drive/folders/1IKEuRULUz-Uvb8ZL8vvYw0Z49aD_Qp_4?usp=sharing) that includes (1) 3 pretrained checkpoints for Tangle, Tangle-Rec and Intra, and (2) pre-extracted slide embeddings for TCGA-BRCA. In addition, we provide a script for downstream evaluation on BRACS and BCNB. 

To run few-shot evaluation:

```bash
# Extract slide embeddings 
python extract_slide_embeddings_from_checkpoint.py --pretrained <PATH_TO_PRETRAINED>
python run_linear_probing.py
```

### Training on all TCGA cohorts

To train Tanglev2 on all TCGA cohorts, use:

```bash
# Train Tangle
source scripts/launch_tanglev2_training.sh
```

Note that we do not provide pre-extracted patch embeddings nor gene expression. 

## Issues 
- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email `gjaume@bwh.harvard.edu` (and cc `avaidya@mit.edu`). 
- Immediate response to minor issues may not be available.

## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{jaume2024transcriptomics,
  title={Transcriptomics-guided Slide Representation Learning in Computational Pathology},
  author={Jaume, Guillaume and Oldenburg, Lukas and Vaidya, Anurag Jayant and Chen, Richard J. and Williamson, Drew FK and Peeters, Thomas and Song, Andrew H. and Mahmood, Faisal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
