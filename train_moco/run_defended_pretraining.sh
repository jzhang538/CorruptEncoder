CUDA_VISIBLE_DEVICES=4,5 python3 main_moco_localized_crop.py \
                        -a resnet18 \
                        --lr 0.06 --batch-size 256 --multiprocessing-distributed \
                        --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
                        --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
                        --dist-url tcp://localhost:10070 \
                        --save-folder-root ./new_ckpt/hunting-dog-defended \
                        --experiment-id exp \
                        ../data/pretraining/hunting-dog_650_0.0_filelist.txt