#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# mirror
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-test.csv

# geo_height
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-test.csv


# polar_sate
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
    --sd_locked=False --accumulate_grad_batches=4 \
    --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
    --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
    --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv \
    --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --text_prompt='' --unconditional_guidance_scale=1.0 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --text_prompt='' --unconditional_guidance_scale=1.0 \
#     --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
