#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH


# polar_sate
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=512 --image_height=256 \
#     --result_dir=results/202310170926/image_log/test \
#     --model_path=results/202310170926/epoch=62-step=17513.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --result_dir=results/202310240632/image_log/test \
#     --model_path=results/202310240632/epoch=62-step=17513.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/lacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --result_dir=results/202310191630/image_log/epoch=62_test \
#     --model_path=results/202310191630/epoch=62-step=17513.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/polar_sate/val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/lacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --result_dir=results/202310201708/image_log/test \
#     --model_path=results/202310201708/epoch=62-step=17513.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/cdte/val.csv
srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
    --num_gpus=4 \
    --config_path=./models/lacldm_v15.yaml \
    --image_width=1024 --image_height=256 \
    --result_dir=results/202310240955/image_log/test \
    --model_path=results/202310240955/epoch=62-step=17513.ckpt \
    --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sat2density/val.csv