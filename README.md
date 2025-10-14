# STGNET:STGNET: Imputation of spatial transcriptome based on imaging techniques using deep generative adversarial and graph convolutional networks
The rapid advancement of spatial transcriptomics provides a unique spatial perspective on tissue structure. Image-based methods like seqFISH and MERFISH offer single-cell resolution but are limited to detecting a restricted number of genes, which may impede a comprehensive view of gene spatial distribution.
Regarding these issues, many methods for enhancing ST data have been proposed based on the similarity between ST cells and reference single-cell RNA sequencing (scRNA-seq) cells. In contrast, we utilize generative adversarial network (GAN) to learn the underlying data distribution of scRNA-seq (SC) data and extract abundant feature information from both SC and ST data respectively. Then, we use a multi-view graph convolutional network (GCN) with an attention mechanism to aggregate node information, thus obtaining accurately enhanced ST data. 
# Description
The input to STGNET is a pre-processed count single cell matrix, with columns representing cells and rows representing genes. It's output is an imputed count matrix with the same dimension. The complete pipeline and the datasets used in the paper is described with the following image.
![STGNET-Pipeline](image/STGNET.png)
# Environment Requirement
1. `R version > 3.5`
2. `Information about the operating environment can be found in the requirement.txt file。`
# Example to run scMultiGAN
1. Data preprocessing process
- `Rscript generate.data.R --expression_matrix_path "raw.txt" --file_suffix "txt" --label_file_path "label.txt"`
Running this code will generate input data for GAN network training and output two parameters required for training on the screen, img_size 和 ncls。demo_data is processed data.
2. Train MultiGAN
- `python pretrain/train_MultiGAN.py`
Here, we mainly obtain a pre trained weight parameter for use in the next stage. Meanwhile, there are many parameters available for modification here.
3. Train MultiGAN_impute
- `python pretrain/train_MultiGAN_impute.py`
When the imputeronly parameter is used, load the imputer model of train_MultiGAN_impute.py to impute expression matrix.
4. Data preprocess
- `python finetune/data_process.py`
This is to generate h5ad data for training purposes.
5. Train finetune model
- `python finetune/train.py`
Effectively utilize the positional information in spatial transcriptome data to fine tune the interpolated data.
