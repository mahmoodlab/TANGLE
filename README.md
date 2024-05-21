# Code for Transcriptomics-guided Slide Representation Learning in Computational Pathology

Welcome to the official GitHub repository of our CVPR 2024 paper, "Transcriptomics-guided Slide Representation Learning in Computational Pathology". This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. 

[[arXiv]](https://arxiv.org/abs/2405.11618) 

![Alt text for the image](support/framework.png "Optional title")

## Highlights
This work introduces the first method for **slide representation learning** using **multimodal pretraining**. Slide representation learning defines a new class of self-supervised methods that aim at extracting information-rich embeddings of histology whole-slide images without using explicit supervision (such as cancer subtype or cancer grade). In this work, we introduce **TANGLE**, a method for Slide + Expression (S+E) pretraining. Conceptually, this method follows the CLIP principle widely employed in Vision-Language models. Here, **we align the slide with its corresponding gene expression profile**. The resulting slide encoder embeds the underlying molecular landscape of the tissue and can, as such, be used for various downstream tasks. In this work, we focus on morphological subtyping of breast and lung cancer and morphological lesion detection in pre-clinical drug safety studies.  

## Code
This repository contains the implementation of **TANGLE**, with step-by-step instructions to pretrain a TANGLE model on the **TCGA-BRCA** (Invasive breast cancer) cohort. The resulting models is evaluated using linear probing (logistic regression) on BCNC for ER/PR/HER2 status prediction. While **TANGLE** paper uses CTransPath and ResNet50-IN patch encoders, we provide instructions and checkpoints when using the UNI patch encoder (Chen et al., Nature Medicine, 2024). 

In addition, we provide a script and checkpoint for pan-cancer **TANGLE** pretraining that aggregates **all** TCGA cohorts into a unified training. This model was not part of TANGLE and is referred to as TANGLEv2. More information will follow. 

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

This code assumes that you have already preprocessed the data, using (1) tissue segmentation, (2) patch coordinate extraction, and (3) patch encoding. In the original paper, we train TANGLE using 2 patch encoder: a ResNet50 features (pretrained on ImageNet) and CTransPath. Here, we use the recently proposed UNI model that leads to better downstream performance. 

- Tissue segmentation, patching, and ResNet50 patch embedding extraction can be done using the [CLAM toolbox](https://github.com/mahmoodlab/CLAM).
- CTransPath patch embedding extraction can be done using their [official implementation](https://github.com/Xiyue-Wang/TransPath). 
- UNI patch embedding extraction can be done following instruction [here](https://github.com/mahmoodlab/UNI). 

Preprocessing the corresponding gene expression profile can be done in several ways. For the TCGA cohorts, we used normalized RNA sequencing data available [here](https://xenabrowser.net/datapages/?dataset=TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443).

### Training on TCGA-BRCA

To simplify reproducing results, we provide a link to a [Drive](https://drive.google.com/drive/folders/1GIJEITf5-7lFKil7Dfi3sSmVFgzh-otv?usp=drive_link) that includes (1) TCGA-BRCA UNI patch embeddings, and (2) the corresponding expression profiles as `pt` files. These need to be downloaded and moved to the base directory in a new dir called `data`. The folder structure should look like:

```
data
|__brca
   |____uni_features
        |__tcga_features
   |____rna
```

To train Tangle (and baselines), use:

```bash
# Train Tangle
source scripts/launch_tangle_training.sh
```

### Training on all TCGA cohorts (Tanglev2)

Tanglev2 training assumes you have TCGA cohorts organized in the following format

```
tcga
|__brca
   |____uni_features
        |__slide_id_0.pt
        |__slide_id_1.pt
   |____molecular_data
        |____normed
          |__slide_id_0.pt
          |__slide_id_1.pt
.
.
.
|__ucec
   |____uni_features
        |__slide_id_0.pt
        |__slide_id_1.pt
   |____molecular_data
        |____normed
          |__slide_id_0.pt
          |__slide_id_1.pt
```

To train Tanglev2 on all TCGA cohorts, use:

```bash
# Train Tangle
source scripts/launch_tanglev2_training.sh
```

Note that due to storage constraints, we cannot provide pre-extracted patch embeddings and gene expression data.  

### Evaluate TANGLE (BRCA-trained) and TANGLEv2 (pancancer-trained) on BCNB molecular status prediction

We provide a link to a [Drive](https://drive.google.com/drive/folders/1IKEuRULUz-Uvb8ZL8vvYw0Z49aD_Qp_4?usp=drive_link) that includes (1) 4 pretrained checkpoints for Tangle-PanCancer, Tangle-BRCA, Tangle-Rec and Intra, and (2) pre-extracted BCNC slide embeddings and evaluation. In addition, we provide two scripts for downstream evaluation on BCNB. 

To run few-shot evaluation:

```bash
# Extract slide embeddings 
python extract_slide_embeddings_from_checkpoint.py --pretrained <PATH_TO_PRETRAINED_MODEL>
python run_linear_probing.py
```

AUC performance of these models is:

|            | |   k=1   |      |  |   k=10  |      |  |   k=25  |      |
|------------|-----|-----|------|------|-----|------|------|-----|------|
|            | ER  | PR  | HER2 | ER   | PR  | HER2 | ER   | PR  | HER2 |
| **Tangle (BRCA)** | 0.681 | 0.579   | 0.514   | 0.826    | 0.752   | 0.651   | 0.847    | 0.77   | 0.664   |
| **Tanglev2 (Pancancer)** | 0.637  | 0.587   | 0.53   | 0.791    | 0.72   | 0.63   | 0.817    | 0.755   | 0.67   |
| **Tangle-Rec** | 0.693   | 0.57   | 0.505   | 0.811    | 0.735   | 0.603   | 0.82    | 0.755   | 0.651   |
| **Intra**  | 0.56   | 0.516   | 0.496   | 0.692    | 0.636   | 0.571   | 0.737    | 0.678   | 0.625   |

### Additional TANGLEv2 evaluation 

AUC performance of TANGLEv2 on internal breast and lung cohorts for morphological and molecular subtyping.

|| k=1 | k=10 | k=all | 
|------------|-----|-----|------|
| MGB Breast subtyping | 0.661  | 0.891  | 0.975 |
| MGB Breast ER | 0.604  | 0.661  | 0.715 |
| MGB Breast PR | 0.645  | 0.638  | 0.651 |
| MGB Breast HER2 | 0.509  | 0.59  | 0.685 |
| MGB Lung subtyping | 0.778  | 0.928  | 0.978 |
| MGB Lung TTF-1 | 0.633  | 0.761  | 0.822 |
| MGB Lung P40 | 0.609  | 0.74  | 0.837 |
| MGB Lung P63| 0.595  | 0.638  | 0.755 |
| MGB Lung napsina | 0.621  | 0.771  | 0.779 |
| MGB Lung CDX-2| 0.60  | 0.535  | 0.612 |
| MGB Lung CD5-6 | 0.628  | 0.794  | 0.889 |

Balanced accuracy performance of TANGLEv2 on eBrains and BRACS. 

|| k=1 | k=10 | k=all | 
|------------|-----|-----|------|
| eBrains fine-grained | 0.309  | 0.647  | 0.672 |
| eBrains coarse       | 0.43  | 0.778  | 0.842 |
| BRACS fine-grained   | 0.172  | 0.266  | 0.344 |
| BRACS coarse         | 0.389  | 0.536  | 0.583 |

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

<img src=docs/joint_logo.png> 
