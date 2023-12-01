#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH


# polar_sate
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=512 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=False --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=512 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/cdte/train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/cdte/val.csv
srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
    --sd_locked=True --accumulate_grad_batches=4 \
    --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
    --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=256 \
    --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sat2density/train.csv \
    --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sat2density/val.csv