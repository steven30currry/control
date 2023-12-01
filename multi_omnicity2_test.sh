#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202310151413/image_log/test \
#     --model_path=results/202310151413/epoch=44-step=10304.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-test.csv
srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
    --num_gpus=4 \
    --config_path=models/macldm_v15.yaml \
    --image_width=1024 --image_height=512 \
    --source_image_width=512 --source_image_height=512 \
    --result_dir=results/202310161554/image_log/test \
    --model_path=results/202310161554/epoch=44-step=10304.ckpt \
    --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-test.csv