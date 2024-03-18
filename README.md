# Code for Transcriptomics-guided Slide Representation Learning in Computational Pathology

Welcome to the official GitHub repository for our CVPR 2024 paper, "Transcriptomics-guided Slide Representation Learning in Computational Pathology". This project was developed by the Mahmood Lab at Harvard Medical School and Brigham and Women's Hospital. 

![Alt text for the image](support/framework.png "Optional title")

## Highlights
This work introduces the first method for slide representation learning using multimodal pretraining. Slide representation learning defines a new class of self-supervised methods that aim at extracting information-rich embeddings of hisotlogy whole-slide images, without using explicit supervision (such as cancer subtype or cancer grade). In this work, we introduce TANGLE, a methods for Slide + Expression (S+E) pretraining. Conceptually, this method follows the CLIP principle widely employed in Vision-Language model. 

Here, we align the slide with its corresponding gene expression profile. The resulting slide encoder embeds the underlying molecular landscape of the tissue, and, as such, can be used for various downstream tasks. In this work, we focus on morphological subtyping of breast and lung cancer, and morphological lesion detection in pre-clinical drug safety studies.  

## Code
This repository contains the implementation of TANGLE, with step-by-step instructions and data to reproduce results on the TCGA-BRCA (Invasive breast cancer) cohort. 

### Installation
<!-- Step-by-step instructions to set up the environment and install necessary dependencies. -->

```bash
# Clone repo
git clone https://github.com/mahmoodlab/TANGLE
cd TANGLE

# Create conda env
conda create -n tangle
conda activate tangle

# Instal dependencies
pip install -r requirements.txt
```

<!-- Instructions on how to run the code, including preparing data, training models, and evaluating results. -->

```bash
# Example placeholder command for training the model
python train_tangle.py --config ADD ARGUMENTS

# Example placeholder command for evaluating the model
python evaluate.py --checkpoint path/to/your/model.ckpt
```

### Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{jaume2024transcriptomics,
  title={Transcriptomics-guided Slide Representation Learning in Computational Pathology},
  author={Jaume, Guillaume and Oldenburg, Lukas and Vaidya, Anurag Jayant and Chen, Richard J. and Williamson, Drew FK and Peeters, Thomas and Song, Andrew H. and Mahmood, Faisal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

