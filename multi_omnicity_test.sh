#!/usr/bin/bash

#SBATCH -o test_omnicity3.log
#SBATCH -J test
#SBATCH -p bigdata
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# python multi_omnicity_test.py \
#     --config_path=models/mcldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202308120545/image_log/test \
#     --model_path=results/202308120545/epoch=84-step=20059.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_multihint_test.csv

# python multi_omnicity_test.py \
#     --config_path=models/ccldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202309011033/image_log/test \
#     --model_path=results/202309011033/epoch=38-step=2573.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_multihint_test.csv

# python multi_omnicity_test.py \
#     --config_path=models/eccldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202309041614/image_log/test \
#     --model_path=results/202309041614/epoch=98-step=6533.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_full_multihint_test.csv

# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=1024 --source_image_height=512 \
#     --result_dir=results/202309201131/image_log/test_epoch=66 \
#     --model_path=results/202309201131/epoch=66-step=4421.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_attn_multihint_test.csv

# python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202309201107/image_log/test_epoch=40.bak \
#     --model_path=results/202309201107/epoch=40-step=2705.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view1-sate_attn_test.csv
# python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202309201118/image_log/test_epoch=47 \
#     --model_path=results/202309201118/epoch=47-step=3167.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view2-sate_attn_test.csv
# python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202309201120/image_log/test_epoch=57 \
#     --model_path=results/202309201120/epoch=57-step=3827.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view3-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202309201120/image_log/test \
#     --model_path=results/202309201120/epoch=57-step=3827.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_view4-sate_attn_test.csv


# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=models/malacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=1024 --source_image_height=512 \
#     --result_dir=results/202310181412/image_log/test \
#     --model_path=results/202310181412/epoch=85-step=5675.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/smalacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=1024 --source_image_height=512 \
#     --result_dir=results/202310190836/image_log/epoch=85_test \
#     --model_path=results/202310190836/epoch=85-step=5675.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/lamacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202310242100/image_log/test \
#     --model_path=results/202310242100/epoch=85-step=5675.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_s2sp_view3-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/lamacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202310271312/image_log/test \
#     --model_path=results/202310271312/epoch=14-step=989.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_view3-sate_attn_test.csv


# omnicity
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202310311815/image_log/test \
#     --model_path=results/202310311815/epoch=85-step=5675.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view4-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/slamacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202310311559/image_log/test \
#     --model_path=results/202310311559/epoch=14-step=989.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_cdte_view3-sate_attn_test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/malacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --source_image_width=512 --source_image_height=512 \
#     --result_dir=results/202311060953/image_log/test \
#     --model_path=results/202311060953/epoch=85-step=5675.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view4-sate_attn_test.csv
srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
    --num_gpus=8 \
    --config_path=models/malacldm_v15.yaml \
    --image_width=1024 --image_height=512 \
    --source_image_width=512 --source_image_height=512 \
    --result_dir=results/202311070109/image_log/test \
    --model_path=results/202311070109/epoch=85-step=5675.ckpt \
    --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_sky-mask_view3-sate_attn_test.csv

# cvusa
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/macldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --source_image_width=256 --source_image_height=256 \
#     --result_dir=results/202311012043/image_log/test \
#     --model_path=results/202311012043/epoch=57-step=16123.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/malacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --source_image_width=256 --source_image_height=256 \
#     --result_dir=results/202310311526/image_log/test \
#     --model_path=results/202310311526/epoch=76-step=21405.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/lamacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --source_image_width=256 --source_image_height=256 \
#     --result_dir=results/202311031127/image_log/test \
#     --model_path=results/202311031127/epoch=57-step=16123.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/slamacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --source_image_width=256 --source_image_height=256 \
#     --result_dir=results/202311021757/image_log/test \
#     --model_path=results/202311021757/epoch=57-step=16123.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/sky_mask-val.csv

# cvusa cdte
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/slamacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --source_image_width=256 --source_image_height=256 \
#     --result_dir=results/202310311622/image_log/test \
#     --model_path=results/202310311622/epoch=76-step=21405.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/cdte-val.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python multi_omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=models/lamacldm_v15.yaml \
#     --image_width=1024 --image_height=256 \
#     --source_image_width=256 --source_image_height=256 \
#     --result_dir=results/202310311408/image_log/test \
#     --model_path=results/202310311408/epoch=76-step=21405.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/cdte-val.csv