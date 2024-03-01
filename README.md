## Official Implementation of CorruptEncoder (CVPR 2024)


## Introduction

This repo is the official implementation of CorruptEncoder in pytorch. 

For any other implementation details, please refer to our paper **Data Poisoning based Backdoor Attacks to Contrastive Learning**. (CVPR 2024) [[Paper](https://arxiv.org/pdf/2211.08229.pdf)]

We provide code implementation of pre-training and evaluating clean/backdoored encoders under our default setting (poisoning ratio: 0.5%; number of reference images: 3; number of support reference images: 5 (optional); pre-training dataset: ImageNet100-A; target downstream task: ImageNet100-B). We also provide pre-trained clean encoder, backdoored encoders of **SSL-backdoor**, **PoisonedEncoder** and **CorruptEncoder(+)** for comparison. Each encoder is pre-trained under the same default setting.

![img](./assets/teasar.png)


## Setup environment

Install the [pytorch](https://pytorch.org/). The latest codes are tested on PyTorch 1.7 and Python 3.6 (also compatible to Pytorch 2.0 and Python 3.10).


## Usage

1. Generate the filelist of a downstream dataset (please specify the path to ImageNet dataset):


		cd get_downstream_dataset
        
        bash quick.sh

2. Generate the filelist of a pre-training dataset (please specify the path to ImageNet dataset):
    

        cd generate-poison

        bash quick_cl.sh

3. Generate a poisoned pre-training dataset and the corresponding filelist (please specify the path to ImageNet dataset):
        
        
        ###
        bash quick.sh

        ### Use CorruptEncoder
        python3 generate_poisoned_images.py --target-class hunting-dog --support-ratio 0

        python3 generate_poisoned_filelist.py --target-class hunting-dog --support-ratio 0

        ### Use CorruptEncoder+
        python3 generate_poisoned_images.py --target-class hunting-dog --support-ratio 0.2

        python3 generate_poisoned_filelist.py --target-class hunting-dog --support-ratio 0.2

4. Pre-train the undefended image encoder on a clean/poisoned pre-training dataset:


        cd train_moco

        bash run_pretraining.sh

5. Pre-train the (localized cropping) defended image encoder on a clean/poisoned pre-training dataset:


        bash run_defended_pretraining.sh

5. Train the linear layer for a downstream classifier:
    

        bash run_linear.sh

6. Evaluate the **CA** (clean accuracy) and **ASR** (attack success rate) of a downstream classifer built based on a clean/backdoored encoder (please specify the target class):


        bash run_test.sh

7. (Optional) We also provide pre-trained clean/backdoored encoders. Download pre-trained encoders from this [[URL](https://drive.google.com/file/d/1N1uFe5UlN8Frh3KsXW4dfmg_Ly4RF-NF/view?usp=sharing)] and put 'ckpt' at the root folder.

        
        CorruptEncoder
        └── train_moco
        └── ckpt
            └── ImageNet100-B-CorruptEncoder
            └── ImageNet100-B-CorruptEncoder+
            └── ...
        └── ...


## Note

1. The object mask is annotated using the tool called labelme: https://github.com/wkentaro/labelme.

2. CorruptEncoder+ can improve the attack stability and performance, but it requires additional reference images.  


## Results

Here we illustrate the expected results of each pre-trained encoder provided in this repo:

| Model | Downstream Task | CA | ASR |
:-: | :-: | :-: | :-:
| Clean | ImageNet100-B | 61.2 | 0.4 |
| CorruptEncoder | ImageNet100-B | 61.6 | 92.9 |
| CorruptEncoder+ | ImageNet100-B | 61.7 | **99.5** |
| PoisonedEncoder | ImageNet100-B | 61.1 | 35.5 |
| SSL-Backdoor | ImageNet100-B | 61.3 | 14.3 |


## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{zhang2022corruptencoder,
  title={CorruptEncoder: Data Poisoning based Backdoor Attacks to Contrastive Learning},
  author={Zhang, Jinghuai and Liu, Hongbin and Jia, Jinyuan and Gong, Neil Zhenqiang},
  journal={arXiv preprint arXiv:2211.08229},
  year={2022}
}
```
