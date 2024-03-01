# Remember to change dataset_path, ckpt_path 


### Use newly pre-trained encoders

CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
                        --arch moco_resnet18 \
                        --weights ./new_ckpt/hunting-dog/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
                        --train_file ../data/imagenet100_B/ds_train.txt \
                        --val_file ../data/imagenet100_B/ds_test.txt \
                        --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ./new_ckpt/hunting-dog-plus/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ./new_ckpt/hunting-dog-defended/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ./new_ckpt/lorikeet/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ./new_ckpt/rottweiler/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ./new_ckpt/komondor/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=7 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ./new_ckpt/clean/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

### Use our pre-trained encoders

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ../ckpt/clean/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear_imageNet100_B

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ../ckpt/ImageNet100-B-CorruptEncoder/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ../ckpt/ImageNet100-B-CorruptEncoder+/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ../ckpt/ImageNet100-B-PoisonedEncoder/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights ../ckpt/ImageNet100-B-SSLBackdoor/checkpoint_0199.pth.tar \
#                         --train_file ../data/imagenet100_B/ds_train.txt \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --linear_layer_dir linear