#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# # hint channels: 3
# srun -p bigdata --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=8 --image_width=512 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/height_train_v3.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/height_test_v3.csv
# medium-view polar sate image
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=False \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/near_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/near_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=False \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --max_epochs=10 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 --drop_prompt_ratio=0.5 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_valid.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 --drop_prompt_ratio=0.1 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 --drop_prompt_ratio=0.1 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_caption_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_caption_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --only_mid_control=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_mask_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_mask_test.csv

# LAControlNet
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_s2sp_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_s2sp_test.csv
srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
    --num_gpus=4 \
    --sd_locked=True --accumulate_grad_batches=8 \
    --config_path=./models/lacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
    --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
    --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_trainval.csv \
    --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_test.csv

# # hint channels: 4
# srun -p bigdata --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --config_path=./models/cldmv2_v15.yaml --model_path=./models/controlv2_sd15_ini.ckpt \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/merged_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/merged_test.csv

# hint channels: 9
# srun -p bigdata --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --config_path=./models/cldmv3_v15.yaml --model_path=./models/controlv3_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=8 --image_width=512 --image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/polar-sate_height_semantic-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/polar-sate_height_semantic-test.csv
# srun -p bigdata --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --config_path=./models/cldmv3_v15.yaml --model_path=./models/controlv3_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/polar-sate_height_semantic-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/polar-sate_height_semantic-test.csvo

# # height estimation
# srun -p bigdata --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_train.py \
#     --config_path=./models/cldm_v15.yaml --model_path=./models/control_sd15_ini.ckpt \
#     --learning_rate=1e-4 --batch_size=8 --image_width=512 --image_height=512 \
#     --text_prompt="a realistic and grayscale height map for satellite image" \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view_height_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view_height_test.csv