#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# medium-view polar sate image
# srun -p bigdata --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --config_path=./models/mcldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/multihint_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/multihint_test.csv

# medium-view geo height
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True \
#     --config_path=./models/mcldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_multihint_train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_multihint_valid.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --drop_context_ratio 0.1 \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/ccldm_v15.yaml --model_path=./models/ccontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_multihint_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_multihint_test.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --drop_context_ratio 0.1 \
#     --sd_locked=True --accumulate_grad_batches=4 \
#     --config_path=./models/eccldm_v15.yaml --model_path=./models/eccontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=4 --image_width=1024 --image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_full_multihint_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_full_multihint_test.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=1024 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_test.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_view1-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view1-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_view2-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view2-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_view3-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view3-sate_attn_test.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/malacldm_v15.yaml --model_path=./models/malacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=1024 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/smalacldm_v15.yaml --model_path=./models/lacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=1024 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_test.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/lamacldm_v15.yaml --model_path=./models/lamacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_s2sp_view3-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_s2sp_view3-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --num_gpus=4 \
#     --sd_locked=True --accumulate_grad_batches=16 \
#     --config_path=./models/lamacldm_v15.yaml --model_path=./models/lamacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_view3-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_view3-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --num_gpus=4 \
#     --sd_locked=True --accumulate_grad_batches=16 \
#     --config_path=./models/slamacldm_v15.yaml --model_path=./models/slamacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_view3-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_view3-sate_attn_test.csv



# cvusa
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=256 --source_image_width=256 --source_image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --num_gpus=4 \
#     --sd_locked=True --accumulate_grad_batches=16 \
#     --config_path=./models/malacldm_v15.yaml --model_path=./models/malacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=256 --source_image_width=256 --source_image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/lamacldm_v15.yaml --model_path=./models/lamacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=256 --source_image_width=256 --source_image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/slamacldm_v15.yaml --model_path=./models/slamacontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=256 --source_image_width=256 --source_image_height=256 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-train.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv



# omnicity
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
#     --sd_locked=True --accumulate_grad_batches=8 \
#     --config_path=./models/macldm_v15.yaml --model_path=./models/mcontrol_sd15_ini.ckpt \
#     --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
#     --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view4-sate_attn_trainval.csv \
#     --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view4-sate_attn_test.csv
srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_train.py \
    --sd_locked=True --accumulate_grad_batches=8 \
    --config_path=./models/malacldm_v15.yaml --model_path=./models/malacontrol_sd15_ini.ckpt \
    --learning_rate=1e-5 --batch_size=2 --image_width=1024 --image_height=512 --source_image_width=512 --source_image_height=512 \
    --train_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view4-sate_attn_trainval.csv \
    --valid_data_file=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view4-sate_attn_test.csv