#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# geo_height-sate_attn
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-test.csv
srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
    --sd_locked=True --accumulate_grad_batches=8 \
    --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
    --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
    --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-train.csv \
    --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-test.csv
