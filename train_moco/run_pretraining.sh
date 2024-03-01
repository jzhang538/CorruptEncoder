CUDA_VISIBLE_DEVICES=4,5 python3 main_moco.py \
                        -a resnet18 \
                        --lr 0.06 --batch-size 256 --multiprocessing-distributed \
                        --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
                        --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
                        --dist-url tcp://localhost:10066 \
                        --save-folder-root ./new_ckpt/hunting-dog \
                        --experiment-id exp \
                        ../data/pretraining/hunting-dog_650_0.0_filelist.txt

# CUDA_VISIBLE_DEVICES=4,5 python3 main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10067 \
#                         --save-folder-root ./new_ckpt/hunting-dog-plus \
#                         --experiment-id exp \
#                         ../data/pretraining/hunting-dog_650_0.2_filelist.txt

# CUDA_VISIBLE_DEVICES=6,7 python3 main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10068 \
#                         --save-folder-root ./new_ckpt/lorikeet \
#                         --experiment-id exp \
#                         ../data/pretraining/lorikeet_650_0.0_filelist.txt

# CUDA_VISIBLE_DEVICES=6,7 python3 main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10069 \
#                         --save-folder-root ./new_ckpt/rottweiler \
#                         --experiment-id exp \
#                         ../data/pretraining/rottweiler_650_0.0_filelist.txt

# CUDA_VISIBLE_DEVICES=6,7 python3 main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10070 \
#                         --save-folder-root ./new_ckpt/komondor \
#                         --experiment-id exp \
#                         ../data/pretraining/komondor_650_0.0_filelist.txt

# CUDA_VISIBLE_DEVICES=4,5 python3 main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10071 \
#                         --save-folder-root ./new_ckpt/clean \
#                         --experiment-id exp \
#                         ../data/pretraining/clean_filelist.txt