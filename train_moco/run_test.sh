# Remember to change dataset_path, ckpt_path and target class
# ImageNet100-B hunting-dog 33 (Default)
# ImageNet100-B lorikeet 6
# ImageNet100-B rottweiler 26
# ImageNet100-B komondor 25

### Use newly pre-trained encoders

CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
                        --arch moco_resnet18 \
                        --evaluate \
                        --eval_data exp \
                        --load_cache \
                        --weights ./new_ckpt/hunting-dog/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
                        --resume ./new_ckpt/hunting-dog/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
                        --val_file ../data/imagenet100_B/ds_test.txt \
                        --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
                        --target_cls 33

# CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ./new_ckpt/hunting-dog-plus/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --resume ./new_ckpt/hunting-dog-plus/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

# CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ./new_ckpt/hunting-dog-defended/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --resume ./new_ckpt/hunting-dog-defended/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

# CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ./new_ckpt/lorikeet/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --resume ./new_ckpt/lorikeet/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 6

# CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ./new_ckpt/rottweiler/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --resume ./new_ckpt/rottweiler/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 26

# CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ./new_ckpt/komondor/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --resume ./new_ckpt/komondor/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 25

# CUDA_VISIBLE_DEVICES=5 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ./new_ckpt/clean/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --resume ./new_ckpt/clean/exp/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

### Use our pre-trained encoders

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ../ckpt/clean/checkpoint_0199.pth.tar \
#                         --resume ../ckpt/clean/linear_imageNet100_B/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ../ckpt/ImageNet100-B-CorruptEncoder/checkpoint_0199.pth.tar \
#                         --resume ../ckpt/ImageNet100-B-CorruptEncoder/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ../ckpt/ImageNet100-B-CorruptEncoder+/checkpoint_0199.pth.tar \
#                         --resume ../ckpt/ImageNet100-B-CorruptEncoder+/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ../ckpt/ImageNet100-B-PoisonedEncoder/checkpoint_0199.pth.tar \
#                         --resume ../ckpt/ImageNet100-B-PoisonedEncoder/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33

# CUDA_VISIBLE_DEVICES=1 python3 eval_linear.py \
#                         --arch moco_resnet18 \
#                         --evaluate \
#                         --eval_data exp \
#                         --load_cache \
#                         --weights ../ckpt/ImageNet100-B-SSLBackdoor/checkpoint_0199.pth.tar \
#                         --resume ../ckpt/ImageNet100-B-SSLBackdoor/linear/checkpoint_0199.pth.tar \
#                         --val_file ../data/imagenet100_B/ds_test.txt \
#                         --val_poisoned_file ../data/imagenet100_B/ds_poisoned_test.txt  \
#                         --target_cls 33